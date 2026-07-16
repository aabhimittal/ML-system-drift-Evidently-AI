# Overview — From Drift Detection to Drift *Correction*

> Part 1 of the series · Next: [02 — Setup & first run](02-setup.md) · See also: [Architecture](architecture.md)

This guide explains the problem this repository solves, the vocabulary you need
(the drift taxonomy), the closed loop the system implements, and where each
concern lives in the codebase.

---

## 1. The problem: model rot

A machine-learning model is a snapshot of the world at training time. The world
keeps moving; the model does not. Slowly (or suddenly) the data flowing through
production stops looking like the data the model learned from, and prediction
quality decays. This is **model rot**, and it happens silently — the model
keeps returning confident predictions while its accuracy quietly collapses.

Most tutorials answer this with *detection*: compute some statistics, render a
report, show a red "drift detected" banner. That is necessary but **not
sufficient**. Detection is only the alarm. The questions that actually matter
in production come after the alarm:

1. **Is this drift the kind that hurts?** Inputs can shift without the model
   getting worse.
2. **What should we do about it?** Full retrain? Reweight recent data? Drop an
   unstable feature? Nothing?
3. **Did the fix actually work?** A retrained model is not automatically a
   better model.
4. **Should the fix be deployed?** Only if it demonstrably beats the incumbent.

This repository implements that entire post-detection path, not just the alarm.

## 2. Drift taxonomy, in plain language

Three distinct things can drift. They have different causes and demand
different responses.

### Covariate (data) drift — the *inputs* change

The distribution of the features shifts, but the underlying relationship
between features and label still holds.

> *Example:* your fraud model was trained when most traffic came from the web;
> a marketing push moves 50% of traffic to mobile. `channel` now looks nothing
> like training data — but a mobile transaction is still fraudulent for the
> same reasons it always was.

In this repo: `DriftSpec.numeric_mean_shift`, `numeric_scale`, and
`categorical_shift` inject exactly this — means move, variances widen,
category probabilities re-weight (see
[03 — Data & training](03-data-and-training.md)).

### Concept drift — the *relationship* changes

The mapping from inputs to label itself changes. The model sees familiar-looking
inputs but the correct answers have moved. This is the dangerous one: yesterday's
model is simply *wrong*, and no amount of input monitoring alone reveals it —
you need fresh labels or a performance signal.

> *Example:* after a pricing change, the same customer profile that used to
> churn now stays. The features look identical; the label logic changed.

In this repo: `DriftSpec.concept_rotation` (a value in `[0, 1]`) rotates the
logistic coefficients that generate the label, so the learned relationship is
progressively decoupled from the new labels.

### Label / prior drift — the *outcome mix* changes

The distribution of the target itself shifts: e.g. the base rate of positives
moves from 30% to 55%. It often accompanies the other two kinds, and it skews
calibrated probabilities and threshold-based decisions.

> *Example:* a recession doubles the default rate. Even a model whose ranking
> is still good now under-predicts risk in absolute terms.

In this repo: the config exposes `drift.target_drift_threshold` (PSI on the
target/prediction distribution), and the Evidently `TargetDriftPreset` report
visualises it.

## 3. The loop this repo implements

```
Detect → Decide → Remediate → Validate → Promote → Monitor
```

| Stage | What happens here | Where |
|-------|-------------------|-------|
| **Detect** | Per-feature KS/PSI (numeric) and Chi²/PSI (categorical) aggregated to a dataset-level verdict; Evidently HTML reports on top | `src/ml_drift/drift/detector.py`, `reports.py` |
| **Decide** | A policy maps drift share + live performance drop to a strategy: `none` · `feature_stabilize` · `reweight` · `retrain` | `src/ml_drift/optimization/strategies.py` |
| **Remediate** | Build the right training set (rolling window / recency-weighted / unstable-features-dropped) and retrain a challenger | `src/ml_drift/optimization/retrainer.py` |
| **Validate** | Champion vs challenger, both scored on a held-out slice of *current* data that neither trained on | `src/ml_drift/optimization/optimizer.py` |
| **Promote** | Challenger replaces `artifacts/models/champion.joblib` only if it wins by `min_improvement` | `optimizer.py` + `pipeline/orchestrator.py` |
| **Monitor** | Every run appended to `artifacts/metrics/monitoring_history.json`; trends in Streamlit | `src/ml_drift/monitoring/dashboard.py`, `app/streamlit_app.py` |

The whole loop runs from one command:

```bash
python scripts/run_pipeline.py --scenario severe
```

and its behaviour is governed entirely by [`config/config.yaml`](../config/config.yaml).

## 4. Repository tour

```
ML-system-drift-Evidently-AI/
├── config/config.yaml              # single source of truth: thresholds & policy
├── src/ml_drift/
│   ├── config.py                   # config loader
│   ├── data/        generate.py    # synthetic data + DriftSpec injection
│   │                loader.py
│   ├── models/      train.py       # ModelBundle: pipeline + metrics + persistence
│   │                predict.py     # score frames for Evidently
│   ├── drift/       detector.py    # native KS / PSI / Chi² core (no deps)
│   │                reports.py     # Evidently AI report suite (optional)
│   ├── optimization/strategies.py  # policy: which remediation to apply
│   │                retrainer.py   # mechanics: build training set + retrain
│   │                optimizer.py   # orchestration + champion/challenger gate
│   ├── monitoring/  dashboard.py   # history logging + trend plot
│   └── pipeline/    orchestrator.py# end-to-end run + CLI (`main`)
├── scripts/                        # generate_data · train_baseline · detect_drift · optimize · run_pipeline
├── app/streamlit_app.py            # interactive dashboard
├── examples/quickstart.py          # ~20-line self-contained demo
├── tests/                          # 28 pytest tests (run offline, no Evidently needed)
├── docs/                           # this guide series
└── .github/workflows/ci.yml        # CI: tests + Evidently report artifacts
```

Reading order for the concerns:

- **Data & drift injection** → `src/ml_drift/data/generate.py` — the `DriftSpec`
  dataclass and the `none()` / `moderate()` / `severe()` presets.
- **Baseline model** → `src/ml_drift/models/train.py` — sklearn `Pipeline`
  wrapped in a serialisable `ModelBundle`.
- **Detection** → `src/ml_drift/drift/detector.py` — pure numpy/scipy, no
  Evidently required for decisions.
- **The core contribution** → `src/ml_drift/optimization/` — policy, retraining
  mechanics, and the champion/challenger gate.
- **Glue** → `src/ml_drift/pipeline/orchestrator.py` — `run_pipeline()` executes
  all seven steps and is also the `ml-drift` CLI entry point.

## 5. Remediation strategies at a glance

The policy in `strategies.py` weighs two signals: *how much of the input
drifted* (drift share, unstable-feature set) and *whether the model is actually
hurting* (drop in the primary metric on freshly-labelled current data).

| Strategy | When the policy picks it | What it does |
|----------|--------------------------|--------------|
| `none` | No dataset-level drift **and** performance drop below `performance_drop_to_retrain` | Keep the champion; log and move on |
| `feature_stabilize` | A *small* set of features (≤ ¼ of them) has very high PSI (`psi_drop_feature_threshold`) but performance still holds | Drop/quarantine the unstable features and retrain a leaner model |
| `reweight` | Moderate drift — above tolerance but below the retrain bar | Retrain on reference + current with recent rows up-weighted (`recent_sample_weight`) |
| `retrain` | Drift share ≥ `min_drift_share_to_retrain` **or** performance drop ≥ `performance_drop_to_retrain` | Full retrain on a rolling window of the most recent data (`rolling_window_size`) |

You can also force any of these via `optimization.strategy` in the config
(`auto` lets the policy decide). Whatever the strategy, the resulting challenger
must still pass the champion/challenger gate before it is promoted — remediation
never blindly replaces the deployed model.

## 6. Where to go next

- [02 — Setup & first run](02-setup.md): install, run the pipeline, read the output.
- [03 — Data & training](03-data-and-training.md): the synthetic data model,
  `DriftSpec`, and the baseline champion.
- [Architecture](architecture.md): component diagram and the seven-step data flow.
