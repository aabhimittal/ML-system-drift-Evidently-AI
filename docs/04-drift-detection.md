# Drift Detection: the Native Statistical Core + Evidently Reports

> Part 4 of the series — prev: [03 Data & Training](03-data-and-training.md) · next: [05 Optimization & Remediation](05-optimization-remediation.md)

This guide covers **Step 3** of the pipeline: deciding, statistically, whether the
"current" production batch has drifted away from the "reference" batch the
champion model was trained on.

The system deliberately has **two layers**:

1. **A native statistical core** (`src/ml_drift/drift/detector.py`) — pure
   NumPy/pandas/SciPy, no Evidently required. This is what actually makes the
   drift *decision* that the remediation policy acts on, and it is what the test
   suite exercises offline.
2. **An Evidently AI report layer** (`src/ml_drift/drift/reports.py`) — an
   *optional* dependency that produces rich, shareable HTML reports on top of the
   same data. If Evidently is not installed, everything still runs.

This split matters in practice: your alerting/remediation logic should not go
down because a heavyweight reporting library changed its API, and your CI should
be able to run the whole decision path in a minimal environment.

---

## 1. The native core: per-feature tests

`detect_dataset_drift(...)` runs one test pair per feature, chosen by dtype:

| Feature kind | Statistical test | Distance metric |
|---|---|---|
| Numeric | Two-sample Kolmogorov–Smirnov (`scipy.stats.ks_2samp`) | PSI over quantile bins |
| Categorical | Chi-square (`scipy.stats.chi2_contingency`) | PSI over category frequencies |

A feature is flagged **drifted** when *either* signal fires:

```
drifted = (p_value < pvalue_threshold) OR (psi >= psi_threshold)
```

Why both? The hypothesis tests are extremely sensitive at large sample sizes
(with 6,000 rows even a tiny, harmless shift can push the p-value below 0.05),
while PSI measures the *magnitude* of the shift. Combining them catches both
"statistically real" and "practically large" changes.

### 1.1 Numeric features: KS + PSI

The two-sample Kolmogorov–Smirnov test compares the empirical CDFs of the
reference and current samples. Its p-value answers: *could these two samples
plausibly come from the same distribution?*

```python
ks_stat, p_value = stats.ks_2samp(ref, cur)
psi = population_stability_index(ref, cur)
drifted = bool(p_value < ks_pvalue_threshold or psi >= psi_threshold)
```

### 1.2 The Population Stability Index (PSI)

PSI measures how much probability mass moved between bins:

```
PSI = Σ over bins  (cur% − ref%) · ln(cur% / ref%)
```

Implementation details from `population_stability_index(reference, current, bins=10)`:

- **Bins come from the reference**: 10 quantile bins are computed on the
  reference sample (`np.quantile` over `np.linspace(0, 1, bins + 1)`), so every
  bin holds ~10% of the reference mass. Duplicate edges (near-constant features)
  are collapsed with `np.unique`; a truly constant feature returns PSI = 0.
- **Outer edges are widened to ±∞** so current values falling outside the
  reference range still land in the tail bins instead of being lost.
- A small epsilon (`1e-6`) is added to both percentage vectors so empty bins do
  not produce `log(0)`.

Rules of thumb (the industry-standard reading, also noted in the docstring):

| PSI | Interpretation |
|---|---|
| `< 0.1` | No meaningful shift |
| `0.1 – 0.2` | Minor shift — watch it |
| `> 0.2` | Major shift — investigate / act |

The default `drift.psi_threshold: 0.2` in [`config/config.yaml`](../config/config.yaml)
therefore flags a feature only at the "major shift" level.

### 1.3 Categorical features: Chi-square + PSI

For categorical columns the detector builds a 2×K contingency table of category
counts (reference row vs current row, one column per category seen in either
sample) and runs `scipy.stats.chi2_contingency`:

```python
table = np.vstack([ref_counts, cur_counts])
stat, p_value, _, _ = stats.chi2_contingency(table + 1e-6)
```

Degenerate tables (a single category, empty columns) are guarded and default to
`p_value = 1.0` (no drift evidence). PSI is computed the same way as for
numerics, but over **category frequencies** instead of quantile bins
(`_categorical_psi`), again with an epsilon for categories absent from one side.

---

## 2. From features to a dataset-level verdict

After every feature is tested, the detector aggregates:

```
drift_share   = n_drifted / n_features
dataset_drift = drift_share >= dataset_drift_share      # default 0.5
```

With the default config, drift in **half or more** of the features declares
dataset-level drift. This share (and the per-feature details) is exactly what
the remediation policy in [docs/05](05-optimization-remediation.md) consumes.

---

## 3. The API

### 3.1 `detect_dataset_drift(...)`

```python
from ml_drift.drift.detector import detect_dataset_drift

result = detect_dataset_drift(
    reference,                      # pd.DataFrame
    current,                        # pd.DataFrame
    features=None,                  # default: shared columns, minus target/prediction cols
    ks_pvalue_threshold=0.05,
    chi2_pvalue_threshold=0.05,
    psi_threshold=0.2,
    dataset_drift_share=0.5,
)
```

When `features` is omitted, the detector compares all columns present in **both**
frames, excluding the label-ish columns `{"target", "prediction",
"prediction_proba"}` — so you can safely pass scored frames.

### 3.2 `FeatureDrift` — one row per feature

```python
@dataclass
class FeatureDrift:
    feature: str        # column name
    kind: str           # "numerical" | "categorical"
    drifted: bool
    psi: float
    test_name: str      # "ks" | "chi2"
    statistic: float    # KS statistic or chi-square statistic
    p_value: float
```

`FeatureDrift.as_dict()` returns the same fields with values rounded to 4
decimals — this is what the Streamlit per-feature table renders.

### 3.3 `DriftResult` — the aggregate verdict

```python
@dataclass
class DriftResult:
    dataset_drift: bool
    drift_share: float
    n_features: int
    n_drifted: int
    features: List[FeatureDrift]
```

Three conveniences you will use constantly:

- `result.drifted_features` — property, names of all drifted features.
- `result.unstable_features(psi_threshold)` — features with `psi >= psi_threshold`.
  The optimizer calls this with `optimization.psi_drop_feature_threshold` (0.5)
  to find candidates for the `feature_stabilize` strategy.
- `result.as_dict()` — JSON-ready dict (`dataset_drift`, `drift_share`,
  `n_features`, `n_drifted`, `drifted_features`, per-feature `features` list),
  used by the CLI printout and the metrics files.

---

## 4. The Evidently layer (`reports.py`)

Evidently is imported inside a `try/except`, setting a module-level flag:

```python
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import (
        ClassificationPreset, DataDriftPreset, DataQualityPreset, TargetDriftPreset,
    )
    EVIDENTLY_AVAILABLE = True
except Exception:      # ImportError or API drift on newer versions
    EVIDENTLY_AVAILABLE = False
```

> **Version pin:** the project uses Evidently's classic `Report` +
> `metric_preset` API, i.e. `evidently>=0.4,<0.5`. Evidently 0.5+ redesigned the
> API; the broad `except Exception` means an incompatible version simply
> disables the report layer instead of crashing the pipeline.

Four report builders wrap the four presets — each returns the saved `Path`, or
`None` when Evidently is unavailable (or required prediction columns are missing):

| Function | Preset | What it shows |
|---|---|---|
| `build_data_drift_report(...)` | `DataDriftPreset` | Per-feature distribution drift |
| `build_target_drift_report(...)` | `TargetDriftPreset` | Shift in target / prediction distribution |
| `build_performance_report(...)` | `ClassificationPreset` | Classification quality on the current batch (needs a `prediction` column in both frames) |
| `build_data_quality_report(...)` | `DataQualityPreset` | Missing values, ranges, cardinality |

A `ColumnMapping` tells Evidently which columns are the target, prediction, and
numeric/categorical features (built by the internal `_column_mapping` helper).

The one-call entry point:

```python
from ml_drift.drift.reports import EVIDENTLY_AVAILABLE, generate_all_reports

reports = generate_all_reports(
    reference_scored, current_scored,
    reports_dir="artifacts/reports",
    target="target", prediction="prediction",
    numeric=champion.numeric_features,
    categorical=champion.categorical_features,
    prefix="severe",                 # filenames become severe_data_drift.html, ...
)
# -> {"data_drift": ".../severe_data_drift.html", "target_drift": ..., 
#     "performance": ..., "data_quality": ...}  (values are None if skipped)
```

`generate_all_reports` **never raises** when Evidently is missing — every value
in the returned dict is simply `None`, and the native detector remains the
source of truth for the drift decision.

---

## 5. Running detection

```bash
python scripts/detect_drift.py            # or: make detect
python scripts/detect_drift.py --no-reports   # native decision only, skip HTML
```

The script loads `data/reference.csv` and `data/current.csv`, runs
`detect_dataset_drift` with the thresholds from config, and prints
`drift.as_dict()` as JSON. If `artifacts/models/champion.joblib` exists it also
scores both frames (so the target-drift and performance reports have a
`prediction` column) and calls `generate_all_reports`.

**HTML reports land in `artifacts/reports/`** (the `paths.reports_dir` config
key). When run via the full pipeline, filenames are prefixed with the scenario,
e.g. `artifacts/reports/severe_data_drift.html`.

---

## 6. Configuration reference (`drift:` block)

From [`config/config.yaml`](../config/config.yaml):

```yaml
drift:
  psi_threshold: 0.2           # PSI >= 0.2  => feature considered drifted
  ks_pvalue_threshold: 0.05    # KS p-value < 0.05 => numeric feature drifted
  chi2_pvalue_threshold: 0.05  # Chi-square p-value < 0.05 => categorical drifted
  dataset_drift_share: 0.5     # >= this share of drifted features => dataset drift
  target_drift_threshold: 0.1  # PSI threshold on target/prediction distribution
```

Tuning intuition:

- **Lower `psi_threshold` / raise p-value thresholds** → more sensitive, more
  alarms (and, downstream, more retrains).
- **`dataset_drift_share`** is your "how much of the world must change" dial.
  0.5 is conservative; monitoring teams often alert earlier (0.3) but *act*
  later.
- These thresholds feed straight into the remediation policy — see
  [docs/05](05-optimization-remediation.md) for how the drift share interacts
  with `optimization.min_drift_share_to_retrain`.

---

## 7. Try it in five lines

```python
from ml_drift.data.generate import DriftSpec, generate_dataset
from ml_drift.drift.detector import detect_dataset_drift

ref = generate_dataset(3000, seed=42, spec=DriftSpec.none())
cur = generate_dataset(3000, seed=43, spec=DriftSpec.severe())
print(detect_dataset_drift(ref, cur).as_dict())
```

You should see most features drifted and `dataset_drift: true`. Next, the
interesting part: what the system *does* about it →
[05 Optimization & Remediation](05-optimization-remediation.md).
