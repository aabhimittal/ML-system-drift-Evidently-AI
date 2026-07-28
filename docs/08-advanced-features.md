# Advanced, Production-Oriented Features

> Part 8/9 of the series — follows [docs/04-drift-detection.md](04-drift-detection.md) and [docs/05-optimization-remediation.md](05-optimization-remediation.md).

The first seven guides built the core loop: **detect → decide → remediate → validate → promote → monitor**. This guide covers a second layer of features that answer the questions an *on-call ML engineer* actually asks in production:

1. [Is this batch even the shape I expect?](#1-data-contract--schema-validation) — schema/data-contract validation
2. [Which drift actually matters to the model?](#2-impact-weighted-drift) — impact-weighted drift
3. [Can I get a warning before labels arrive?](#3-unsupervised-prediction-drift) — prediction drift
4. [Is drift hiding inside a segment?](#4-segmented-drift) — per-slice detection
5. [Are PSI and KS telling me the whole story?](#5-extra-distances-js-divergence--normalized-wasserstein) — JS divergence + Wasserstein
6. [With many features, how do I avoid alarm floods?](#6-benjaminihochberg-fdr-correction) — Benjamini–Hochberg FDR
7. [Is retraining actually worth the money?](#7-costvalue-aware-remediation-gate) — the economic gate

Everything below runs offline (no Evidently required) and is demonstrated end-to-end by [`examples/advanced_features.py`](../examples/advanced_features.py) — run it with `make advanced` or `python examples/advanced_features.py`.

---

## 1. Data-contract / schema validation

**What.** `src/ml_drift/drift/schema.py` infers a `SchemaContract` from the reference (training) frame — one `ColumnContract` per column recording dtype kind, numeric min/max, tolerated null rate, and (for low-cardinality categoricals) the allowed category set — then enforces it against each incoming batch.

**Why.** In real pipelines the nastiest incidents are not subtle statistical drift but *hard schema breaks*: a renamed column, a numeric field that started arriving as a string, a null rate that jumped from 0% to 40%, out-of-range values from an upstream bug, or a brand-new categorical level. That is why the pipeline runs this check **before** any statistical drift test — a KS test on a broken column is meaningless.

**How.** The one-call convenience wrapper infers and validates in one step:

```python
from ml_drift.drift.schema import validate_batch

report = validate_batch(reference, batch, ignore=["target"])
print(report.ok, report.n_errors, report.n_warnings)
for v in report.violations:
    print(v.severity, v.kind, v.column, v.detail)
```

Or hold on to the contract and reuse it across batches:

```python
from ml_drift.drift.schema import SchemaContract

contract = SchemaContract.infer(reference, ignore=["target"],
                                range_slack=0.0, null_slack=0.02, max_categories=50)
report = contract.validate(batch, out_of_range_error_frac=0.05)
```

Every `SchemaViolation` has a `kind` and a `severity`:

| `kind` | Meaning | Typical severity |
|---|---|---|
| `missing_column` | A required column is absent | **error** |
| `unexpected_column` | A column not in the contract appeared | warning |
| `dtype_change` | e.g. numeric column arriving as strings | **error** if >50% of values un-parseable, else warning |
| `null_spike` | Null rate exceeds contract (reference rate + `null_slack`) | warning |
| `out_of_range` | Numeric values outside the learned min/max | **error** if the out-of-range fraction exceeds `out_of_range_error_frac` (systematic upstream problem), else warning (a few outliers) |
| `unseen_category` | New categorical level not seen in training | warning |

The rule of thumb: **errors** break scoring or silently corrupt it; **warnings** are tolerable but worth alerting. `report.ok` is `True` iff there are zero errors.

Config: `advanced.enable_schema_validation` (see [section 8](#8-how-it-all-surfaces-in-the-pipeline)).

---

## 2. Impact-weighted drift

**What.** `src/ml_drift/drift/advanced.py` ranks each feature by `impact = PSI × model importance` and sums it into a single `total_impact` score.

**Why.** A PSI of 0.8 on a feature the model barely uses is noise; a PSI of 0.3 on its most important feature is a fire. Plain drift shares (see [docs/04](04-drift-detection.md)) treat every feature equally — impact weighting distinguishes *dangerous* drift from *benign* drift.

**How.** `model_feature_importances(bundle)` extracts per-original-feature importances from a trained `ModelBundle`. The pipeline one-hot-expands categoricals, so importances of the expanded columns (e.g. `cat__region_north`) are **summed back to the original feature** (`region`). It works with tree models (`feature_importances_`) and linear models (`|coef_|`), and normalises to sum to 1.

```python
from ml_drift.drift.advanced import model_feature_importances, impact_weighted_drift
from ml_drift.drift.detector import detect_dataset_drift

drift = detect_dataset_drift(reference, current)
importances = model_feature_importances(champion)
iwd = impact_weighted_drift(drift, importances)

print(iwd.total_impact, iwd.top_feature)   # ImpactWeightedDrift
for r in iwd.rankings[:5]:                 # ImpactRanking: feature, psi, importance, impact
    print(r.feature, r.psi, r.importance, r.impact)
```

Real output from `examples/advanced_features.py` under the severe scenario:

```
total impact score = 1.64   top feature = score
feature        psi  importance   impact
score          ...         ...      ...   <- highest PSI × importance product
...
```

`score` drifts hard **and** the model relies on it heavily — that combination, not raw PSI alone, is what makes it the top remediation priority.

Config: `advanced.enable_impact_weighting`.

---

## 3. Unsupervised prediction drift

**What.** `detect_prediction_drift(reference_scores, current_scores)` monitors the *model's own output distribution* — the predicted probabilities — and returns a `PredictionDrift` with `drifted`, `psi`, `js_divergence`, `ks_statistic`, `ks_pvalue`, and `mean_shift`.

**Why.** In most industrial systems ground-truth labels arrive days or weeks late (or never). You cannot compute ROC-AUC on today's traffic, but you *can* watch the score distribution: if the model suddenly scores everything differently, the world has changed. It is the earliest available warning signal.

**How — and the p-value trap.** The drift decision is deliberately based on **effect size** — PSI and the *magnitude* of the KS statistic — **not** the KS p-value:

```python
drifted = psi >= psi_threshold or ks_stat >= ks_stat_threshold   # 0.2 and 0.1 by default
```

At production batch sizes the KS p-value rejects on statistically-tiny, operationally-irrelevant differences — the classic *"everything is significant at n = 100k"* trap. A `ks_stat_threshold=0.1` means "at some point the two CDFs differ by at least 10 percentage points of probability mass", which is a real, size-independent effect. The p-value is still reported for reference.

```python
from ml_drift.drift.advanced import detect_prediction_drift
from ml_drift.models.predict import predict_proba

pdrift = detect_prediction_drift(predict_proba(champion, reference),
                                 predict_proba(champion, current))
print(pdrift.drifted, pdrift.psi, pdrift.ks_statistic, pdrift.mean_shift)
```

Config: `advanced.enable_prediction_drift`, `advanced.prediction_psi_threshold`.

---

## 4. Segmented drift

**What.** `segmented_drift(reference, current, segment_column, ...)` runs the full detector (see [docs/04](04-drift-detection.md)) independently within each level of a segment column and returns a `SegmentedDrift` whose `worst_segment` property gives `(segment, drift_share)` for the worst slice.

**Why.** Aggregate drift can look calm while one region, device type, or customer tier is severely drifted — the calm majority masks the burning pocket. Per-slice detection surfaces it.

**How.**

```python
from ml_drift.drift.advanced import segmented_drift

seg = segmented_drift(reference, current, "region", min_segment_size=100)
for name, res in seg.segments.items():          # each value is a full DriftResult
    print(name, res.drift_share, res.dataset_drift)
print(seg.worst_segment)                        # e.g. ('south', 0.89)
```

Segments smaller than `min_segment_size` in *either* frame are skipped — too little data for a reliable verdict beats a noisy one.

Config: `advanced.segment_column` (set to `null` to disable), `advanced.min_segment_size`.

---

## 5. Extra distances: JS divergence & normalized Wasserstein

**What.** `src/ml_drift/drift/stats.py` adds two distances that are now computed per feature and reported in `FeatureDrift.as_dict()` alongside PSI and the test statistic:

- **Jensen–Shannon divergence** (`jensen_shannon_divergence`, plus a `_categorical` variant) — symmetric and **bounded in [0, 1]** (base-2), so it is safe to threshold and to average across features. PSI, by contrast, is unbounded and blows up on rare bins.
- **Normalized Wasserstein** (`normalized_wasserstein`) — earth-mover distance divided by the reference standard deviation, making it **scale-free**: a shift is comparable between a feature measured in dollars and one measured in milliseconds. (It is 0.0 for categorical features.)

**Why.** No single distance is enough. KS is shape-sensitive but its p-value is sample-size dependent; PSI is intuitive but unbounded; JS gives a bounded, comparable severity score; Wasserstein tells you *how far* the mass moved, not just that it moved.

```python
from ml_drift.drift.stats import jensen_shannon_divergence, normalized_wasserstein

js   = jensen_shannon_divergence(ref_values, cur_values, bins=20)   # in [0, 1]
wass = normalized_wasserstein(ref_values, cur_values)               # in reference-std units

# Or just read them off any detector result:
for f in detect_dataset_drift(reference, current).features:
    print(f.feature, f.psi, f.js_divergence, f.wasserstein)
```

---

## 6. Benjamini–Hochberg FDR correction

**What.** `benjamini_hochberg(pvalues, alpha)` in `stats.py` implements the classic false-discovery-rate procedure; the detector applies it when you pass `correction="bh"`.

**Why.** With hundreds of monitored features, raw per-feature p-values at α = 0.05 will flag ~5% of them *even with no true drift* — a permanent flood of false alarms that trains the on-call engineer to ignore the pager. BH controls the expected *fraction of flagged features that are false alarms* instead of the per-test error rate.

**How.**

```python
from ml_drift.drift.detector import detect_dataset_drift

result = detect_dataset_drift(reference, current, correction="bh", fdr_alpha=0.05)
```

After the per-feature tests run, the detector recomputes each feature's verdict as `BH-rejected OR psi >= psi_threshold` — the PSI effect-size check is kept regardless, so a large practical shift is never suppressed by the correction.

Config: `drift.correction` (`none` | `bh`) and `drift.fdr_alpha` in [`config/config.yaml`](../config/config.yaml). Recommended whenever the feature count is large.

---

## 7. Cost/value-aware remediation gate

**What.** `src/ml_drift/optimization/policy.py` adds an **economic gate** on top of the champion/challenger quality gate from [docs/05](05-optimization-remediation.md). `expected_value_of_retraining(metric_gain, value_per_metric_point, retrain_cost)` converts a metric gain into money and nets off the cost; `should_promote(...)` combines both gates and returns a `PromotionVerdict` (`promote`, `reason`, `expected_value`).

**Why.** Retraining is not free — compute, review time, deployment risk. A challenger that wins by +0.001 AUC passes a naive quality gate but may not be worth a $5,000 deployment cycle. Industrially you retrain when the *expected value* of the improvement outweighs its cost.

**How.** Two gates, applied in order:

1. **Quality gate** (always, unless `champion_challenger=False`): improvement ≥ `min_improvement` and strictly positive.
2. **Economic gate** (only if `value_per_metric_point` is set): `improvement × value_per_metric_point − retrain_cost` must be positive.

```python
from ml_drift.optimization.policy import should_promote

# Challenger improves ROC-AUC 0.62 -> 0.70 (+0.08); a full 1.0 AUC gain is worth $10,000.
for cost in (50, 5000):
    v = should_promote(0.62, 0.70, value_per_metric_point=10000, retrain_cost=cost)
    print(cost, v.promote, v.reason)
```

Output from `examples/advanced_features.py`:

```
retrain_cost=$50    -> promote=True   (quality + economic gates passed (EV +750.00))
retrain_cost=$5000  -> promote=False  (economic gate: expected value -4200.00 does not cover retrain cost 5000.00)
```

Same model, same improvement — the *economics* flip the decision. The gate is wired into `optimize_after_detection` (`src/ml_drift/optimization/optimizer.py`) and hence into the full pipeline, so a run's promotion decision automatically respects it.

Config: `optimization.value_per_metric_point` (leave `null` to disable the economic gate) and `optimization.retrain_cost`.

---

## 8. How it all surfaces in the pipeline

All of the above is orchestrated by `src/ml_drift/pipeline/orchestrator.py` and controlled by the `advanced:` section of [`config/config.yaml`](../config/config.yaml):

```yaml
advanced:
  enable_schema_validation: true
  enable_impact_weighting: true
  enable_prediction_drift: true
  prediction_psi_threshold: 0.2
  segment_column: region     # null to disable
  min_segment_size: 100
```

A pipeline run (`make pipeline` or `python scripts/run_pipeline.py --scenario severe`) now prints an extra block:

```
=== ADVANCED ANALYTICS ===
  schema        : ok=True (errors=0, warnings=2)
  impact drift  : total=1.64 top_feature=score
  pred. drift   : drifted=True (psi=..., js=..., mean_shift=...)  [unsupervised]
  worst segment : region=... (share=...)
```

and the same data lands in the `analytics` block of the run JSON (`artifacts/metrics/<scenario>_run.json`), next to the `drift` and `optimization` blocks — so dashboards and CI can consume it. Analytics are deliberately best-effort: a failure in an analytics step never breaks the run.

To see every feature in isolation with printed output, run:

```bash
make advanced            # or: python examples/advanced_features.py
```

**Next:** [docs/09-industrial-edge-cases.md](09-industrial-edge-cases.md) — how the detector stays sane on pathological production batches.
