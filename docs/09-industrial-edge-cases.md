# Industrial Edge Cases & Robustness

> Part 9/9 of the series — follows [docs/08-advanced-features.md](08-advanced-features.md); the detector itself is introduced in [docs/04-drift-detection.md](04-drift-detection.md).

A drift detector that works on clean tutorial data and crashes at 3 a.m. on a malformed production batch is worse than no detector at all — it takes the whole monitoring pipeline down exactly when something upstream broke. This guide documents how `detect_dataset_drift` (`src/ml_drift/drift/detector.py`) is hardened so that **every pathological batch still yields a sane verdict**: it never raises, and it never manufactures a false "drift!" alarm out of missing or degenerate data.

Why does this matter so much? Because the failure modes below are not exotic — they are the *routine* output of real data platforms:

- an upstream team renames or drops a column without telling you;
- a serialization hop (JSON, CSV, a message queue) silently turns numbers into strings;
- a feed dies and starts emitting nulls, or a division-by-zero emits `inf`;
- a batch job produces zero rows, or one;
- a new product/country/device code appears as an unseen categorical level.

If any of these raises an exception inside `detect_dataset_drift`, the *entire* detect → remediate loop from [docs/05](05-optimization-remediation.md) stops — no verdict, no remediation, no monitoring record. The design rule throughout is therefore:

> **Degrade gracefully, never crash; when there is nothing to compare, say "not drifted" rather than guessing.**

Note the second half: the safe default for a degenerate comparison is *no alarm*. An empty batch or an all-NaN column is a **pipeline** incident, not statistical drift, and raising a drift alarm for it would both mis-diagnose the problem and erode trust in real drift alarms. The schema-contract layer ([docs/08 §1](08-advanced-features.md#1-data-contract--schema-validation)) exists precisely to catch those pipeline incidents by name.

Each case below is locked in by a dedicated test in [`tests/test_edge_cases.py`](../tests/test_edge_cases.py) — the test names are cited inline so you can jump straight to the executable specification.

---

## The quick-reference table

| Edge case | What the detector does | Verdict |
|---|---|---|
| Column missing from current batch | Only columns present in **both** frames are tested; the missing column is skipped | No crash; column absent from results |
| Extra column in current batch | Not part of the reference → not tested | No crash; column absent from results |
| Empty current batch (0 rows) | Every per-feature test sees an empty sample and returns a neutral result | `dataset_drift=False`, no false alarm |
| Single-row current batch | Statistics computed on what is there | No crash; `drift_share` is a valid float |
| All-NaN numeric column | NaNs dropped → empty sample → nothing to compare | `drifted=False` for that feature |
| Constant reference column | KS/PSI undefined; falls back to a mean-shift check (`test_name="constant-ref"`) | `drifted=True` iff current moved off the constant |
| Unseen categorical level | Categories cast to strings; the union of levels forms the contingency table | Normal verdict, no exception |
| Numeric column arriving as strings | Coerced with `pd.to_numeric(errors="coerce")`; un-parseable values become NaN and are dropped | Tested on the parseable values |
| ±inf values | Non-finite values dropped before any statistic | Finite PSI/KS, no crash |
| Single-class `target` column | `target` (and `prediction`, `prediction_proba`) excluded from feature drift by default | Never tested as a feature |
| Identical frames | All tests pass trivially | `dataset_drift=False`, `drift_share=0.0` |
| Extreme disjoint drift (e.g. +100 shift) | Distances saturate | `drifted=True`, JS divergence > 0.5 |
| Many features, no true drift | `correction="bh"` FDR control | BH flags no more features than raw p-values |

---

## Case by case

### Schema mismatch: missing and extra columns

When `features` is not passed explicitly, the detector builds its feature list as *columns present in both frames*, minus the label columns:

```python
skip = {"target", "prediction", "prediction_proba"}
features = [c for c in reference.columns if c in current.columns and c not in skip]
```

So a column dropped upstream (`test_missing_column_in_current_is_ignored`) is simply not tested, and a new column added upstream (`test_extra_column_in_current_is_ignored`) is likewise ignored — statistical drift is only meaningful on shared columns. If you *want* to be alerted about the schema change itself, that is exactly what the data-contract layer is for: `validate_batch` reports `missing_column` as an **error** and `unexpected_column` as a warning ([docs/08 §1](08-advanced-features.md#1-data-contract--schema-validation)). Contract first, statistics second.

### Empty and single-row current batches

An empty current batch (a stalled upstream job, a bad partition filter) gives every test a zero-length sample. Each per-feature helper checks for this and returns a neutral `FeatureDrift` (`drifted=False`, `p_value=1.0`) instead of raising — an empty batch is a *pipeline* problem, not evidence of drift (`test_empty_current_batch`). A single-row batch is statistically almost as useless, but it must not crash either; the detector returns a valid result with a float `drift_share` (`test_single_row_current_batch`).

### All-NaN numeric column

A feed that went dark often keeps the column but fills it with nulls. NaNs are dropped before computing statistics, which leaves an empty sample — the same neutral path as an empty batch, so the feature is reported as not drifted (`test_all_nan_numeric_column`). No comparison → no alarm.

### Constant reference column

If a reference feature has zero range (`np.ptp(ref) == 0`), it has *no distribution to drift from* — KS and quantile-binned PSI are undefined. The detector falls back to a mean-shift check, reported with `test_name="constant-ref"`: the feature is drifted iff the current values differ from the constant. A moved constant is still caught (`test_constant_reference_column` asserts `drifted=True` when the reference is pinned at 3.0 and the current varies). Relatedly, `population_stability_index` itself returns 0.0 for constant inputs (`test_psi_handles_constant`).

### Unseen categorical level

A new category (a new product code, a new country) is the most common "soft" schema change. `_detect_categorical` casts both sides to strings and builds its contingency table over the **union** of levels from both frames, so an unseen level is handled like any other category and contributes to Chi²/PSI naturally (`test_unseen_category_does_not_crash`). The schema layer separately raises an `unseen_category` warning if you want an explicit alert.

### Numeric column arriving as strings

Serialization changes (JSON re-encoding, a CSV export, a type-lossy hop through a queue) routinely turn numerics into strings. When the reference column is numeric but the current one is not, the detector coerces:

```python
if not pd.api.types.is_numeric_dtype(cur_col):
    cur_col = pd.to_numeric(cur_col, errors="coerce")
```

Parseable values are recovered and tested; un-parseable ones become NaN and are dropped downstream (`test_numeric_column_arrives_as_string`). The contract layer will still flag the underlying `dtype_change` so the root cause gets fixed.

### Infinities

`±inf` from an upstream division-by-zero would poison quantiles, means, and histograms. Numeric samples are filtered with `np.isfinite` before any statistic, so PSI/KS remain finite and the rest of the (finite) sample is still tested (`test_infinities_are_handled`).

### Single-class target

If the label collapses to a single class, that is a labeling-pipeline incident — but it cannot crash *feature* drift detection, because `target` is in the default `skip` set and is never tested as a feature (`test_single_class_target_column_not_tested_by_default`). Target/prediction drift is monitored separately (see [docs/04](04-drift-detection.md) and the unsupervised prediction-drift check in [docs/08 §3](08-advanced-features.md#3-unsupervised-prediction-drift)).

### Sanity anchors: identical frames and extreme drift

Two calibration tests pin down both ends of the scale:

- **Identical frames** → zero drift, exactly: `dataset_drift=False`, `drift_share == 0.0` (`test_identical_frames_no_drift`). A detector with any false-positive floor on identical data is untrustworthy.
- **Completely disjoint distributions** (current = reference + 100) → the feature must be flagged, and its bounded JS divergence must exceed 0.5, i.e. the distance metrics saturate as expected (`test_extreme_disjoint_drift`).

### BH correction reduces false alarms

On two *i.i.d.* draws from the same distribution (no true drift), raw per-feature p-values will occasionally reject by chance. `test_bh_correction_reduces_false_alarms` runs the detector twice on such a pair — `correction="none"` vs `correction="bh"` — and asserts BH flags **no more** features than the raw thresholds. See [docs/08 §6](08-advanced-features.md#6-benjaminihochberg-fdr-correction) for the method and the `drift.correction` / `drift.fdr_alpha` config keys.

---

## Try it yourself

Every behavior above is easy to reproduce interactively. A few examples, adapted directly from the test suite:

```python
import numpy as np
import pandas as pd
from ml_drift.drift.detector import detect_dataset_drift

rng = np.random.default_rng(0)
ref = pd.DataFrame({
    "num": rng.normal(0, 1, 400),
    "cat": rng.choice(["a", "b", "c"], 400),
    "target": rng.integers(0, 2, 400),
})
cur = ref.copy()
```

**Feed it garbage — get a verdict, not a traceback:**

```python
# 1. Numeric column arrives as strings (schema drift on the wire)
bad = cur.copy()
bad["num"] = bad["num"].astype(str)
detect_dataset_drift(ref, bad)            # coerced via pd.to_numeric, tested normally

# 2. All-NaN column (dead upstream feed)
bad = cur.copy()
bad["num"] = np.nan
r = detect_dataset_drift(ref, bad)
next(f for f in r.features if f.feature == "num").drifted   # False — no false alarm

# 3. Infinities from an upstream division-by-zero
bad = cur.copy()
bad.loc[bad.index[:10], "num"] = np.inf
r = detect_dataset_drift(ref, bad)
np.isfinite(next(f for f in r.features if f.feature == "num").psi)   # True

# 4. Constant reference column
ref2 = ref.copy(); ref2["num"] = 3.0
r = detect_dataset_drift(ref2, cur)
f = next(f for f in r.features if f.feature == "num")
f.test_name, f.drifted                    # ('constant-ref', True) — moved off the constant
```

**And the calibration anchors:**

```python
detect_dataset_drift(ref, ref.copy()).drift_share    # 0.0 — identical frames, zero drift

far = cur.copy(); far["num"] = far["num"] + 100
f = next(f for f in detect_dataset_drift(ref, far).features if f.feature == "num")
f.drifted, f.js_divergence > 0.5                     # (True, True) — saturated distance
```

---

## Operational guidance: alarm vs. incident

Graceful degradation does not mean these situations are *fine* — it means the drift verdict stays trustworthy while you fix them. A practical triage:

| Symptom in the drift output | Likely root cause | Where to look first |
|---|---|---|
| A previously-tested feature vanished from `features` | Column dropped or renamed upstream | Schema report: `missing_column` **error** ([docs/08 §1](08-advanced-features.md#1-data-contract--schema-validation)) |
| `n_features` fine but one feature suddenly `drifted=False` with `p_value=1.0` | Column went all-NaN or empty | Schema report: `null_spike` warning; upstream feed health |
| `test_name == "constant-ref"` appears | Reference feature was constant (or a stale reference snapshot) | Reference-data build job |
| Categorical feature drifts with new levels in play | New upstream category | Schema report: `unseen_category` warning |
| Drift verdicts unchanged but schema errors present | Dtype change absorbed by coercion | Fix the producer; coercion is a safety net, not a contract |

The pattern: the **detector** keeps the statistical verdict sane; the **schema contract** tells you *why* the batch was weird. Run both — the pipeline does ([docs/08 §8](08-advanced-features.md#8-how-it-all-surfaces-in-the-pipeline)).

---

## Testing philosophy

The suite now contains **75 tests** across `tests/` (data generation, native drift statistics, schema contracts, advanced analytics, remediation strategies, the champion/challenger and economic gates, pipeline smoke tests — and the edge cases above). Three principles:

1. **Offline first.** Every test runs without Evidently installed. The native core (`detector.py`, `stats.py`, `schema.py`, `advanced.py`) makes all decisions; Evidently only adds HTML reports.
2. **Graceful handling, not absence of crashes by luck.** Each edge-case test feeds the detector a deliberately pathological batch and asserts a *specific* sane behavior — the right verdict, a finite statistic, the right column skipped — not merely "it did not raise".
3. **Both ends of the scale are pinned.** Zero-drift and saturated-drift anchors keep threshold tuning honest: sensitivity changes can never silently introduce false positives on identical data or false negatives on disjoint data.

Run it:

```bash
make test          # or: PYTHONPATH=src pytest -q
```

**Previous:** [docs/08-advanced-features.md](08-advanced-features.md) · **Series start:** [docs/01-overview.md](01-overview.md)
