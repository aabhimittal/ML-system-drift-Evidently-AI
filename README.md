# ML System Data Drift — Detection **and Post-Detection Optimization** via Evidently AI

An end-to-end, reproducible MLOps reference project that does not stop at
*detecting* data drift — it **acts on it**. Once Evidently AI (and a native
statistical core) flag that production data has drifted, an automated
**optimization / remediation** layer decides what to do, retrains a challenger,
and promotes it only if it beats the incumbent on the new data.

> **Detect → Decide → Remediate → Validate → Promote → Monitor**

```
          ┌─────────────┐   ┌──────────────┐   ┌───────────────────────┐   ┌──────────────┐
  data →  │  Baseline   │ → │    Drift     │ → │  Post-Detection       │ → │  Monitoring  │
          │   model     │   │  detection   │   │  optimization         │   │  history     │
          │ (champion)  │   │ Evidently +  │   │ strategy → retrain →   │   │  + Streamlit │
          │             │   │ native core  │   │ champion/challenger    │   │  dashboard   │
          └─────────────┘   └──────────────┘   └───────────────────────┘   └──────────────┘
```

Most drift tutorials end at the red "drift detected" banner. In production that
is only the *alarm*. The valuable part — the part this repo implements — is the
**closed loop** that turns a drift alarm into a corrected, validated model.

---

## Why this project exists

A deployed model silently rots as the world changes (covariate drift) or as the
input→label relationship changes (concept drift). Detecting drift is necessary
but not sufficient. This project demonstrates the *optimization step after
detection*:

| Stage | What most demos do | What this repo adds |
|-------|--------------------|---------------------|
| Detect | Show an Evidently report | + native KS/PSI/Chi² core so decisions run offline |
| Decide | (nothing) | A **policy** that picks a remediation strategy from drift + live metrics |
| Remediate | (manual retrain) | Automated **retrain / reweight / feature-stabilize** |
| Validate | (trust the retrain) | **Champion/challenger gate** — promote only if it actually wins |
| Monitor | (one-off) | Append every run to a **history** + trend dashboard |

### The result, in one run

Under a severe drift scenario the champion's ROC-AUC collapses; the system
detects it, retrains, and recovers performance automatically:

```
Dataset drift: True (7/9 features, share=0.78)
Remediation:   retrain — broad drift and/or performance degradation beyond threshold
Champion  roc_auc on current : 0.464     ← the deployed model is now barely better than chance
Challenger roc_auc on current: 0.746     ← remediation recovers performance
Promoted challenger: True                ← champion/challenger gate approves the swap
```

---

## Quickstart (60 seconds)

```bash
# 1. Install (core stack; Evidently is optional but recommended)
pip install -r requirements.txt          # or: make install

# 2. Run the whole loop on a simulated "severe drift" scenario
python scripts/run_pipeline.py --scenario severe          # or: make pipeline

# 3. See the smallest possible example, no config needed
python examples/quickstart.py
```

Prefer the `Makefile`:

```bash
make data       # generate reference + current (drifted) datasets
make train      # train the baseline champion
make detect     # drift detection + Evidently HTML reports
make optimize   # post-detection remediation + champion/challenger
make pipeline   # all of the above, end to end
make test       # run the test suite
make app        # launch the Streamlit dashboard
```

> **No Evidently installed?** Everything still runs. The native drift core
> (`src/ml_drift/drift/detector.py`) makes the detect→remediate decision;
> Evidently only adds the rich HTML reports. `pip install "evidently>=0.4,<0.5"`
> to enable them.

---

## Step-by-step: how the end-to-end system is built

This is the condensed walkthrough. Each step links to a full guide in [`docs/`](docs/).

### Step 1 — Generate reproducible data with *controllable* drift → [docs/03](docs/03-data-and-training.md)
Instead of downloading a dataset (which you cannot make drift on demand), we
generate one. `src/ml_drift/data/generate.py` builds a labelled classification
dataset (6 numeric + 3 categorical features, binary `target` from a logistic
model) and can inject two kinds of drift via `DriftSpec`:
- **Covariate drift** — numeric means/scales shift, categorical probabilities re-weight.
- **Concept drift** — the input→label relationship rotates, so the old model is simply wrong.

Three ready scenarios: `DriftSpec.none()`, `.moderate()`, `.severe()`.

### Step 2 — Train the baseline "champion" model → [docs/03](docs/03-data-and-training.md)
`src/ml_drift/models/train.py` fits a scikit-learn `Pipeline`
(`StandardScaler` + `OneHotEncoder` → `RandomForestClassifier` by default) and
saves a serialisable `ModelBundle` (pipeline + column layout + metrics).

### Step 3 — Detect drift (native core + Evidently) → [docs/04](docs/04-drift-detection.md)
`src/ml_drift/drift/detector.py` runs per-feature tests — **KS** + **PSI** for
numeric, **Chi²** + **PSI** for categorical — and aggregates them into a
dataset-level verdict (drift when the share of drifted features crosses a
threshold). `src/ml_drift/drift/reports.py` layers **Evidently AI** presets on
top (`DataDriftPreset`, `TargetDriftPreset`, `ClassificationPreset`,
`DataQualityPreset`) to emit shareable HTML.

### Step 4 — Optimize *after* detection (the core contribution) → [docs/05](docs/05-optimization-remediation.md)
`src/ml_drift/optimization/` turns the alarm into action:
1. **Policy** (`strategies.py`) picks a strategy from drift share + performance drop:
   `none` · `feature_stabilize` · `reweight` · `retrain`.
2. **Mechanics** (`retrainer.py`) assembles the right training set (rolling
   window / recency-weighted / unstable-features-dropped) and retrains a challenger.
3. **Orchestration** (`optimizer.py`) evaluates champion vs challenger on a
   **held-out slice of the current data** and promotes the challenger only if it
   wins by `min_improvement` — a true champion/challenger gate.

### Step 5 — Monitor over time → [docs/06](docs/06-monitoring-dashboard.md)
Every run appends a record (drift share, metrics, strategy, promotion) to
`artifacts/metrics/monitoring_history.json`. The **Streamlit** app
(`app/streamlit_app.py`) runs the pipeline interactively and renders drift
tables, the Evidently reports, and the trend chart.

### Step 6 — Automate in CI/CD → [docs/07](docs/07-cicd-and-deployment.md)
`.github/workflows/ci.yml` runs the test suite on 3.9/3.11, smoke-tests the
quickstart, and (in a separate job) installs Evidently and uploads the generated
HTML reports as build artifacts.

---

## Repository layout

```
ML-system-drift-Evidently-AI/
├── config/config.yaml              # single source of truth: thresholds & policy
├── src/ml_drift/
│   ├── config.py                   # config loader
│   ├── data/        generate.py    # synthetic data + DriftSpec injection
│   │                loader.py
│   ├── models/      train.py       # ModelBundle: pipeline + metrics + persistence
│   │                predict.py     # score frames for Evidently
│   ├── drift/       detector.py    # native KS / PSI / Chi² core (no deps) + edge-case hardening
│   │                stats.py       # JS divergence, normalized Wasserstein, Benjamini-Hochberg FDR
│   │                advanced.py    # impact-weighted / prediction / segmented drift
│   │                schema.py      # data-contract validation (schema breaks)
│   │                reports.py     # Evidently AI report suite (optional)
│   ├── optimization/strategies.py  # policy: which remediation to apply
│   │                retrainer.py   # mechanics: build training set + retrain
│   │                policy.py      # cost/value-aware promotion gate
│   │                optimizer.py   # orchestration + champion/challenger gate
│   ├── monitoring/  dashboard.py   # history logging + trend plot
│   └── pipeline/    orchestrator.py# end-to-end run + CLI (`main`)
├── scripts/                        # generate_data · train_baseline · detect_drift · optimize · run_pipeline
├── app/streamlit_app.py            # interactive dashboard
├── examples/                       # quickstart.py + advanced_features.py demos
├── tests/                          # 75 pytest tests (run offline, no Evidently needed)
├── docs/                           # step-by-step guides (01–09 + architecture)
└── .github/workflows/ci.yml        # CI: tests + Evidently report artifacts
```

## Advanced / production-oriented features

Beyond the core detect→remediate loop, the system includes features aimed at real industrial deployments (see [docs/08](docs/08-advanced-features.md)):

| Feature | Module | Why it matters |
|---------|--------|----------------|
| **Data-contract validation** | `drift/schema.py` | Catches hard schema breaks (missing/renamed columns, dtype changes, null spikes, out-of-range values, unseen categories) *before* statistical drift — the #1 cause of production incidents. |
| **Impact-weighted drift** | `drift/advanced.py` | Ranks drift by `PSI × model feature-importance`, so a big shift in an ignored feature ranks below a small shift in a pivotal one. |
| **Unsupervised prediction drift** | `drift/advanced.py` | Watches the model's own score distribution as an early-warning signal when labels are delayed/missing. Uses **effect size**, not p-values, to avoid the "everything is significant at large n" trap. |
| **Segmented drift** | `drift/advanced.py` | Runs detection per slice (e.g. per region) so a severe pocket isn't masked by a calm average. |
| **JS divergence + normalized Wasserstein** | `drift/stats.py` | Bounded, scale-free distances that complement PSI/KS. |
| **Benjamini–Hochberg FDR correction** | `drift/stats.py` | Controls false drift alarms when testing hundreds of features (`drift.correction: bh`). |
| **Cost/value-aware remediation gate** | `optimization/policy.py` | Promotes a challenger only when the expected value of its improvement exceeds the retraining cost. |

These surface automatically in the pipeline under `=== ADVANCED ANALYTICS ===` and in the run JSON's `analytics` block; toggle them in the config `advanced:` section. Run `make advanced` (or `python examples/advanced_features.py`) for a live demo.

## Configuration

All behaviour is driven by [`config/config.yaml`](config/config.yaml). The knobs
you will actually tune:

| Key | Meaning |
|-----|---------|
| `drift.psi_threshold` | PSI above which a feature is "drifted" (0.2 = moderate) |
| `drift.dataset_drift_share` | share of drifted features that declares dataset drift |
| `optimization.strategy` | `auto` (policy decides) or force `retrain`/`reweight`/`feature_stabilize`/`none` |
| `optimization.min_drift_share_to_retrain` | drift share that triggers a full retrain |
| `optimization.performance_drop_to_retrain` | metric drop that triggers a retrain |
| `optimization.champion_challenger` | if true, only promote a challenger that wins |

## Testing

```bash
make test          # or: PYTHONPATH=src pytest -q
```

The suite (75 tests) covers data generation, the native drift metrics, model
training/persistence, every remediation strategy, the champion/challenger and
cost-aware gates, the advanced analytics (schema, impact-weighting, prediction &
segmented drift), a dedicated **industrial edge-case matrix** (empty/constant/
all-NaN columns, schema mismatch, unseen categories, infinities, dtype coercion),
and full-pipeline smoke tests. It runs **without Evidently installed**.

## Documentation index

| Guide | Contents |
|-------|----------|
| [docs/01-overview.md](docs/01-overview.md) | Problem, drift taxonomy, system tour |
| [docs/02-setup.md](docs/02-setup.md) | Install, environment, first run |
| [docs/03-data-and-training.md](docs/03-data-and-training.md) | Data generation, drift injection, baseline model |
| [docs/04-drift-detection.md](docs/04-drift-detection.md) | KS/PSI/Chi², Evidently presets, thresholds |
| [docs/05-optimization-remediation.md](docs/05-optimization-remediation.md) | Strategies, retraining, champion/challenger |
| [docs/06-monitoring-dashboard.md](docs/06-monitoring-dashboard.md) | History logging, Streamlit app |
| [docs/07-cicd-and-deployment.md](docs/07-cicd-and-deployment.md) | CI, scheduling, productionisation |
| [docs/08-advanced-features.md](docs/08-advanced-features.md) | Schema contracts, impact-weighted drift, prediction drift, FDR, cost gate |
| [docs/09-industrial-edge-cases.md](docs/09-industrial-edge-cases.md) | Robustness to pathological batches + the edge-case test matrix |
| [docs/architecture.md](docs/architecture.md) | Component diagram + data flow |

## License

See [LICENSE](LICENSE).
