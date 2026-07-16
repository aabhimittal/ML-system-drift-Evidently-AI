# Data Generation & Baseline Training

> Part 3 of the series · Prev: [02 — Setup](02-setup.md) · Next: [04 — Drift detection](04-drift-detection.md)

This guide covers the first two layers of the system: the synthetic dataset
with *controllable* drift (`src/ml_drift/data/generate.py`) and the baseline
"champion" model (`src/ml_drift/models/train.py`).

---

## 1. Why synthetic data?

Drift work has a chicken-and-egg problem: to test a drift detector (and,
crucially, the remediation that follows), you need data whose drift you can
*control*. A downloaded dataset drifts however it happens to drift — you can't
dial covariate drift up while holding concept drift at zero, and you can't
rerun the experiment with a different severity.

Generating the data solves this:

- **Reproducible** — every batch is seeded (`numpy.random.default_rng(seed)`),
  so a run today matches a run next month.
- **Controllable** — each kind of drift is an explicit knob on a `DriftSpec`,
  so you know the ground truth the detector should find.
- **Offline** — no downloads, no credentials; CI and the test suite run
  anywhere.

## 2. The data schema

Every generated batch is a pandas `DataFrame` with 10 columns:

| Column | Type | Reference distribution |
|--------|------|------------------------|
| `amount` | numeric | Normal(50, 20) |
| `tenure` | numeric | Normal(24, 12) |
| `latency` | numeric | Normal(120, 40) |
| `score` | numeric | Normal(0, 1) |
| `balance` | numeric | Normal(1000, 500) |
| `age` | numeric | Normal(40, 12) |
| `region` | categorical | north 0.30 · south 0.30 · east 0.20 · west 0.20 |
| `channel` | categorical | web 0.55 · mobile 0.35 · store 0.10 |
| `device` | categorical | ios 0.40 · android 0.45 · desktop 0.15 |
| `target` | binary (0/1) | logistic model over the features |

The label is not random: `target` is drawn from a **logistic model** — fixed
coefficients over the standardized numeric features plus per-level offsets for
each categorical — so a classifier can genuinely *learn* the reference
relationship, and concept drift can genuinely *break* it. The feature name
lists are exported as `NUMERIC_FEATURES` and `CATEGORICAL_FEATURES` from
`ml_drift.data.generate` (the copies in `config.yaml` are informative only).

## 3. `DriftSpec` — drift as an explicit specification

```python
@dataclass
class DriftSpec:
    numeric_mean_shift: Dict[str, float]            # per-feature shift, in SIGMA units
    numeric_scale: Dict[str, float]                 # per-feature std multiplier
    categorical_shift: Dict[str, Dict[str, float]]  # category -> additive prob weight
    concept_rotation: float = 0.0                   # 0..1, rotates the label model
```

- **`numeric_mean_shift`** — additive shift in units of the feature's own
  standard deviation: `{"amount": 1.5}` moves `amount` up by 1.5σ (i.e. by
  1.5 × 20 = 30 in raw units). Missing features default to no shift.
- **`numeric_scale`** — multiplies the standard deviation: `> 1` widens the
  distribution, `< 1` narrows it.
- **`categorical_shift`** — additive weights applied to the reference category
  probabilities before renormalisation, e.g.
  `{"channel": {"mobile": 0.35, "web": -0.20}}` re-weights traffic toward
  mobile. Pure covariate drift.
- **`concept_rotation`** — magnitude in `[0, 1]` of **concept drift**. `0`
  keeps the reference input→label relationship; `1` fully rotates the logistic
  coefficients toward an alternative vector, so *which inputs drive the label*
  changes while the inputs themselves may look the same.

### The three presets

| Preset | Covariate drift | Concept drift | Intended outcome |
|--------|-----------------|---------------|------------------|
| `DriftSpec.none()` | none | 0.0 | Sanity baseline: detector should stay quiet |
| `DriftSpec.moderate()` | shifts `amount`/`latency`/`score` means, widens `balance`/`latency`, re-weights `channel`/`device` | 0.25 | Realistic "production has drifted" — policy typically reweights or retrains |
| `DriftSpec.severe()` | large shifts on `amount`/`latency`/`score`/`age`, wide `balance`/`latency`/`amount`, re-weights `region`/`channel` | 0.6 | Should force a full retrain and a champion collapse |

These are exactly the `--scenario none|moderate|severe` options on the CLI.

## 4. Generating data in code

```python
from ml_drift.data.generate import DriftSpec, generate_dataset, make_reference_and_current

# A single labelled batch (no drift => reference distribution):
reference = generate_dataset(n_samples=6000, seed=42, spec=DriftSpec.none())

# A drifted batch with a custom spec:
spec = DriftSpec(
    numeric_mean_shift={"amount": 2.0},         # +2 sigma on amount
    numeric_scale={"latency": 1.5},             # 50% wider latency
    categorical_shift={"channel": {"mobile": 0.3, "web": -0.2}},
    concept_rotation=0.4,                       # substantial concept drift
)
current = generate_dataset(n_samples=6000, seed=43, spec=spec)

# Or the one-call convenience — clean reference + drifted current:
reference, current = make_reference_and_current(
    n_samples=6000, seed=42, current_spec=DriftSpec.severe()
)
```

Notes on `make_reference_and_current(n_samples=6000, seed=42, current_spec=None)`:
it returns `(reference_df, current_df)`, uses `seed + 1` for the current batch
(so sampling noise differs even with a no-op spec), and defaults `current_spec`
to `DriftSpec.moderate()` when you pass `None`.

### From the command line

```bash
python scripts/generate_data.py --scenario moderate    # or: make data
# Reference (6000 rows) -> data/reference.csv
# Current   (6000 rows, scenario=moderate) -> data/current.csv
```

Row count and seed come from `data.n_samples` and `project.random_seed` in
[`config/config.yaml`](../config/config.yaml).

## 5. The baseline champion model

`src/ml_drift/models/train.py` wraps everything the rest of the system needs
into one serialisable artifact.

### The sklearn pipeline

```
ColumnTransformer
├── StandardScaler                       → numeric features
└── OneHotEncoder(handle_unknown="ignore") → categorical features
        ↓
RandomForestClassifier   (default; config: n_estimators=200, max_depth=12, min_samples_leaf=5)
```

`build_estimator(model_type, params)` also supports `gradient_boosting` and
`logistic_regression` — switch via `model.type` in the config. Feature types
are *inferred* from dtypes (`_infer_feature_types`), so the pipeline adapts if
you plug in a different dataset.

### `ModelBundle`

```python
@dataclass
class ModelBundle:
    pipeline: Pipeline                 # fitted preprocessing + estimator
    numeric_features: List[str]
    categorical_features: List[str]
    target: str
    model_type: str
    metrics: Dict[str, float]          # held-out validation metrics
```

Prediction, drift reporting, and retraining all consume this single object, so
the column layout travels with the fitted pipeline.

### Training

```python
from ml_drift.models.train import train_model, save_model, load_model

bundle = train_model(
    frame,                     # DataFrame including the target column
    target="target",
    model_type="random_forest",
    params={"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5},
    test_size=0.25,
    seed=42,
    sample_weight=None,        # optional np.ndarray, one weight per row
)
print(bundle.metrics)  # {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ..., 'roc_auc': ...}
```

Key behaviours:

- Data is split with a stratified `train_test_split` (`test_size`, `seed`);
  `bundle.metrics` is evaluated on the held-out slice via `evaluate(...)` and
  contains **accuracy, precision, recall, f1, roc_auc** (ROC-AUC uses
  `predict_proba` and is `nan` if only one class is present).
- **`sample_weight`** is how the *reweight* remediation strategy works: pass an
  array of per-row weights and they are split alongside the data and forwarded
  to the estimator as `model__sample_weight`, giving recent/current rows more
  influence during retraining.

### Persistence

```python
path = save_model(bundle, "artifacts/models/champion.joblib")  # joblib.dump, mkdir -p
bundle = load_model(path)   # raises FileNotFoundError with a helpful message if missing
```

### From the command line

```bash
python scripts/train_baseline.py       # or: make train
# Trained random_forest champion -> artifacts/models/champion.joblib
# Validation metrics:
# { "accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "roc_auc": ... }
```

The script loads `data/reference.csv` if it exists (run `make data` first) and
otherwise generates a fresh reference batch from the config.

Next: [04 — Drift detection](04-drift-detection.md) — the native KS/PSI/Chi²
core and the optional Evidently report suite. For the big picture, see
[Architecture](architecture.md).
