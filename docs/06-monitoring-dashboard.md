# Monitoring Over Time: History Log + Streamlit Dashboard

> Part 6 of the series — prev: [05 Optimization & Remediation](05-optimization-remediation.md) · next: [07 CI/CD & Deployment](07-cicd-and-deployment.md)

Detection and remediation are single-run events. Monitoring is what turns them
into a **time series**: is drift getting worse batch over batch? How often does
the policy retrain? Do promoted challengers actually hold their gains?

This layer is deliberately minimal — a JSON file plus two renderers
(`src/ml_drift/monitoring/dashboard.py` and `app/streamlit_app.py`). It is the
smallest useful version of an Evidently monitoring dashboard, with no external
service required.

---

## 1. The monitoring history file

Every pipeline run (with monitoring logging enabled) appends **one record** to
`artifacts/metrics/monitoring_history.json` — the path comes from
`monitoring.history_file` in [`config/config.yaml`](../config/config.yaml).

The record is assembled in the pipeline orchestrator
(`src/ml_drift/pipeline/orchestrator.py`) from the drift result and the
optimization outcome:

```json
{
  "scenario": "severe",
  "drift_share": 0.7778,
  "dataset_drift": true,
  "n_drifted": 7,
  "strategy": "retrain",
  "champion_metric": 0.464,
  "challenger_metric": 0.746,
  "promoted": true,
  "primary_metric": "roc_auc"
}
```

Field by field:

| Field | Source | Meaning |
|---|---|---|
| `scenario` | CLI/app selection | which drift scenario generated the current batch |
| `drift_share` | `DriftResult.drift_share` | fraction of features flagged drifted |
| `dataset_drift` | `DriftResult.dataset_drift` | the dataset-level verdict |
| `n_drifted` | `DriftResult.n_drifted` | count of drifted features |
| `strategy` | `RemediationDecision.strategy` | remediation chosen (`none`/`feature_stabilize`/`reweight`/`retrain`) |
| `champion_metric` | `OptimizationOutcome` | champion's primary metric on the held-out eval slice |
| `challenger_metric` | `OptimizationOutcome` | challenger's metric on the same slice (`null` when strategy was `none`) |
| `promoted` | `OptimizationOutcome.promoted` | whether the champion/challenger gate approved the swap |
| `primary_metric` | `monitoring.primary_metric` | the metric name the numbers refer to (default `roc_auc`) |

### 1.1 The three functions in `dashboard.py`

```python
from ml_drift.monitoring.dashboard import append_history, load_history, plot_history

append_history("artifacts/metrics/monitoring_history.json", record)  # returns Path
history = load_history("artifacts/metrics/monitoring_history.json")  # list[dict]
plot_history("artifacts/metrics/monitoring_history.json",
             "artifacts/metrics/history.png")                        # Path | None
```

- **`load_history(path)`** — returns the list of records; a missing file,
  invalid JSON, or a non-list payload all return `[]` (monitoring must never
  crash the pipeline).
- **`append_history(path, record)`** — creates parent directories, loads the
  existing list, appends, and rewrites the file (`indent=2`, `default=str` so
  odd types serialise).
- **`plot_history(path, out_path)`** — renders the trend chart to a PNG and
  returns its path, or `None` if the history is empty or matplotlib is missing.

### 1.2 The trend chart

`plot_history` uses matplotlib with the **headless `Agg` backend**
(`matplotlib.use("Agg")`), so it works in CI and on servers without a display.
It draws a dual-axis figure over run index 1..N:

- Left axis: **drift share** as translucent orange bars (fixed 0–1 scale).
- Right axis: **champion metric** as a blue solid line with circles, and — if
  any run produced one — the **challenger metric** as a green dashed line with
  squares.

One glance answers the key question: *when drift spikes, does the challenger
line recover what the champion line lost?*

---

## 2. The Streamlit app (`app/streamlit_app.py`)

The interactive front end runs the whole pipeline on demand and renders every
layer of the system:

```bash
make app
# or
streamlit run app/streamlit_app.py
```

What you get:

- **Sidebar controls** — pick a drift scenario (`none` / `moderate` / `severe`,
  from the pipeline's `SCENARIOS` registry) and a remediation strategy
  (`auto` / `retrain` / `reweight` / `feature_stabilize` / `none`, overriding
  the configured strategy for that run), then hit **Run pipeline**.
- **Headline metrics** — four cards: dataset-drift verdict, drift share,
  chosen strategy, and whether a model was promoted.
- **Post-detection optimization panel** — the policy's `reason`, champion vs
  challenger primary metric (with the improvement as a delta), and the
  champion's performance drop on the current batch.
- **Per-feature drift table** — a `st.dataframe` of every `FeatureDrift.as_dict()`
  row: feature, kind, drifted, PSI, test, statistic, p-value.
- **Embedded Evidently reports** — when `EVIDENTLY_AVAILABLE` is true, each
  generated HTML report is read from disk and embedded inline via
  `st.components.v1.html(...)`. When Evidently is not installed the app shows an
  info box and the native drift verdict above remains authoritative.
- **Monitoring history** — always shown at the bottom: a `st.line_chart` of
  `drift_share` over runs plus the full history table. Runs launched from the
  app call `run_pipeline(cfg, scenario=..., log_monitoring=True)`, so every
  click adds a row.

---

## 3. Building a real time series

The history only becomes interesting after several runs. Simulate a "life of a
model" by running scenarios in sequence:

```bash
python scripts/run_pipeline.py --scenario none      --log-monitoring
python scripts/run_pipeline.py --scenario none      --log-monitoring
python scripts/run_pipeline.py --scenario moderate  --log-monitoring
python scripts/run_pipeline.py --scenario severe    --log-monitoring
python scripts/run_pipeline.py --scenario moderate  --log-monitoring
```

> Note the flag: the `run_pipeline.py` CLI only appends to the history when
> `--log-monitoring` is passed (the Streamlit app and the `run_pipeline(...)`
> Python API log by default).

Then open the dashboard (`make app`) or render the static chart:

```python
from ml_drift.monitoring.dashboard import plot_history
plot_history("artifacts/metrics/monitoring_history.json",
             "artifacts/metrics/history.png")
```

Reading the resulting series:

- `none` runs — low drift share, `strategy: none`, no challenger points.
- `moderate` runs — drift share rises; the policy typically reweights or
  retrains depending on the measured performance drop.
- `severe` runs — drift bars near the top, champion line craters, challenger
  line recovers, `promoted: true`.

Each run also writes a full per-run summary (drift details + optimization
outcome + report paths) to `artifacts/metrics/<scenario>_run.json`, so the
compact history rows can always be traced back to the complete evidence.

---

## 4. Beyond this repo: Evidently's monitoring stack

This JSON-file-plus-chart approach is intentionally the *minimal* monitoring
loop: append-only, dependency-free, easy to inspect and diff. Natural next steps
if you outgrow it:

- **Evidently Workspace / monitoring UI** — Evidently ships its own workspace
  concept where each run's `Report` is added as a snapshot and browsed in a web
  UI with panels over time. The `generate_all_reports(...)` call in
  [reports.py](04-drift-detection.md#4-the-evidently-layer-reportspy) is the
  natural integration point: add each report to a workspace instead of (or in
  addition to) saving standalone HTML.
- **A collector service** — instead of batch CSVs, stream production predictions
  to a collector and compute drift on rolling windows.
- **A real metrics store** — swap the JSON file for a database or a
  Prometheus/Grafana pair once multiple models or environments need the same
  view. The record schema above maps directly onto labeled metrics.

The important part is already in place: every run emits a small, structured,
append-only record. Whatever backend you choose later, the loop's telemetry
contract doesn't change.

Next: wiring this loop into CI/CD and production schedules →
[07 CI/CD & Deployment](07-cicd-and-deployment.md).
