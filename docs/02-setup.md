# Setup & First Run

> Part 2 of the series · Prev: [01 — Overview](01-overview.md) · Next: [03 — Data & training](03-data-and-training.md)

This guide gets you from a clean machine to a full end-to-end pipeline run —
data generation, drift detection, remediation, and the champion/challenger
decision — in a few minutes.

---

## 1. Prerequisites

- **Python 3.9 or newer** (CI tests 3.9 and 3.11).
- `pip` and, optionally, `make`.
- No GPU, no external services, no internet at runtime — the whole system runs
  offline on synthetic data.

Check your interpreter:

```bash
python3 --version    # should print 3.9+
```

## 2. Create a virtual environment

Always work in a venv so the project's pinned ranges don't fight your system
packages:

```bash
cd ML-system-drift-Evidently-AI
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

## 3. Install — two paths

### Path A: requirements file (quickest)

```bash
pip install -r requirements.txt
```

This installs the core ML stack (numpy, pandas, scipy, scikit-learn, joblib),
PyYAML, matplotlib + streamlit for the dashboard, pytest, **and** Evidently.

### Path B: editable install (recommended for development)

```bash
pip install -e ".[dev]"        # or: make install
```

This installs the package from `pyproject.toml` in editable mode with the
`dev` extras (pytest, pytest-cov), and registers a console script so you can
run the pipeline as:

```bash
ml-drift --scenario severe
```

### Evidently is optional

The decision-making core is **native** (`src/ml_drift/drift/detector.py`,
pure KS/PSI/Chi² on numpy/scipy), so *everything* — detection, remediation,
champion/challenger, the tests — runs **without Evidently installed**.
Evidently only adds the rich, shareable HTML reports. To enable them:

```bash
pip install "evidently>=0.4,<0.5"
```

The version pin matters: the code uses Evidently's classic
`Report` / `metric_preset` API, which changed in 0.5.

## 4. First run

### The full pipeline

```bash
python scripts/run_pipeline.py --scenario severe     # or: make pipeline
```

This regenerates data, trains (or loads) the champion, detects drift, writes
Evidently reports (if installed), runs post-detection optimization, and prints
a summary like:

```
=== DRIFT DETECTION ===
  dataset_drift : True  (share=0.78, 7/9 features)
  drifted       : amount, latency, score, age, balance, region, channel

=== POST-DETECTION OPTIMIZATION ===
  strategy      : retrain  (broad drift and/or performance degradation beyond threshold)
  roc_auc      : champion=0.464 challenger=0.746  (Δ=0.282)
  decision      : PROMOTED challenger
```

That is the whole story in four lines: under severe drift the deployed model's
ROC-AUC collapses to near chance (0.464), the policy chooses `retrain`, the
challenger recovers to 0.746, and the champion/challenger gate approves the
swap. (Exact numbers vary slightly with library versions but the pattern —
collapse, recover, promote — is stable and seeded.)

### The minimal example

```bash
python examples/quickstart.py
```

A ~20-line, config-free script that does the same loop in memory:

```
Champion validation: {'accuracy': ..., 'roc_auc': ..., ...}

Dataset drift: True (7/9 features, share=0.78)
Drifted features: [...]

Remediation: retrain - broad drift and/or performance degradation beyond threshold
Champion roc_auc on current : 0.464
Challenger roc_auc on current: 0.746 (Δ +0.282)
Promoted challenger: True
```

## 5. Makefile targets

| Target | Command it runs | What it does |
|--------|-----------------|--------------|
| `make install` | `pip install -e ".[dev]"` | Editable install + dev extras |
| `make data` | `python scripts/generate_data.py` | Generate reference + current (drifted) datasets |
| `make train` | `python scripts/train_baseline.py` | Train the baseline champion |
| `make detect` | `python scripts/detect_drift.py` | Drift detection + Evidently HTML reports |
| `make optimize` | `python scripts/optimize.py` | Post-detection remediation + champion/challenger |
| `make pipeline` | `python scripts/run_pipeline.py` | Everything, end to end (default scenario: `moderate`) |
| `make monitor` | `python scripts/run_pipeline.py --log-monitoring` | Pipeline run + append to monitoring history |
| `make test` | `PYTHONPATH=src python -m pytest` | Run the 28-test suite |
| `make app` | `streamlit run app/streamlit_app.py` | Launch the interactive dashboard |
| `make clean` | `rm -rf artifacts data/*.csv ...` | Remove generated artifacts |

Useful CLI flags on `scripts/run_pipeline.py`:

```bash
python scripts/run_pipeline.py --scenario none|moderate|severe   # pick drift scenario
python scripts/run_pipeline.py --no-reports                      # skip Evidently reports
python scripts/run_pipeline.py --no-regenerate                   # reuse existing data/*.csv
python scripts/run_pipeline.py --log-monitoring                  # append run to history
python scripts/run_pipeline.py --config path/to/config.yaml      # alternate config
```

## 6. What a run leaves behind

```
data/reference.csv                        # clean reference batch
data/current.csv                          # drifted current batch
artifacts/models/champion.joblib          # deployed model (updated on promotion)
artifacts/reports/*.html                  # Evidently reports (if installed)
artifacts/metrics/<scenario>_run.json     # full auditable run summary
artifacts/metrics/monitoring_history.json # one record per run (with --log-monitoring)
```

All paths come from the `paths:` section of
[`config/config.yaml`](../config/config.yaml).

## 7. Troubleshooting

### `ModuleNotFoundError: No module named 'ml_drift'`

The package lives under `src/`. Three ways to make it importable:

1. **Editable install** (best): `pip install -e ".[dev]"`.
2. **PYTHONPATH**: `PYTHONPATH=src python -m pytest` — this is exactly what
   `make test` does, and `pyproject.toml` also sets `pythonpath = ["src"]`
   for pytest.
3. **Use the scripts**: everything in `scripts/` and `examples/` prepends
   `src/` to `sys.path` itself, so `python scripts/run_pipeline.py` works with
   no install at all.

### `evidently` fails to install or is slow

Evidently pulls a fairly large dependency tree (plotly and friends), so the
install can take a while on slow connections. Remember: **you don't need it**
to run detection, remediation, or the tests. Skip it, or install it later with
`pip install "evidently>=0.4,<0.5"`. Without it the pipeline prints
`Evidently reports: skipped (evidently not installed)` and continues.

### Evidently installed but reports fail

Check the version — `evidently>=0.4,<0.5` is required. Version 0.5+ removed
the classic `Report`/`metric_preset` API this project uses.

### `FileNotFoundError: Model artifact not found`

You ran a step that needs the champion (`scripts/detect_drift.py`,
`scripts/optimize.py`) before training it. Run `make train` first, or just use
`make pipeline`, which trains it for you.

Next: [03 — Data & training](03-data-and-training.md) — how the synthetic data
and the baseline champion actually work.
