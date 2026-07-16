# CI/CD and Deployment: Automating the Loop

> Part 7 of the series — prev: [06 Monitoring & Dashboard](06-monitoring-dashboard.md) · see also: [architecture.md](architecture.md)

The final step is making the detect→remediate loop run without a human at the
keyboard. This guide walks through the GitHub Actions workflow that ships with
the repo, then sketches how to take the reference implementation toward
production: scheduled drift checks, a real prediction service, a model registry,
and alerting/deploy gates.

---

## 1. The CI workflow (`.github/workflows/ci.yml`)

The workflow triggers on every push (all branches) and on pull requests into
`main`. It has **two jobs**, mirroring the project's two-layer design (native
core vs Evidently reports).

### 1.1 `test` — the offline core, on a version matrix

```yaml
strategy:
  fail-fast: false
  matrix:
    python-version: ["3.9", "3.11"]
```

For each Python version (3.9 and 3.11) it:

1. Checks out the repo (`actions/checkout@v4`) and sets up Python with pip
   caching (`actions/setup-python@v5`).
2. Installs **only the light scientific stack** — deliberately *not*
   `requirements.txt`:

   ```bash
   pip install numpy pandas scipy scikit-learn joblib PyYAML pytest pytest-cov
   ```

   Evidently is intentionally absent: this job proves the claim from
   [docs/04](04-drift-detection.md) that the native drift core and the entire
   decision path run without it.
3. Runs the test suite with coverage:

   ```bash
   PYTHONPATH=src pytest -q --cov=src/ml_drift --cov-report=term-missing
   ```

4. Smoke-tests the minimal example: `python examples/quickstart.py`.

`fail-fast: false` means a failure on one Python version doesn't cancel the
other — you see the full compatibility picture in one run.

### 1.2 `evidently-reports` — the full pipeline, artifacts uploaded

A separate job (Python 3.11 only) proves the Evidently integration end-to-end:

1. Installs the **full** stack: `pip install -r requirements.txt` (which
   includes the `evidently>=0.4,<0.5` pin).
2. Runs the whole loop under the worst-case scenario:

   ```bash
   python scripts/run_pipeline.py --scenario severe
   ```

   This generates data, trains the champion, detects drift, emits the four
   Evidently HTML reports, and runs the remediation + champion/challenger gate.
3. Uploads the reports as a build artifact:

   ```yaml
   - uses: actions/upload-artifact@v4
     with:
       name: evidently-reports
       path: artifacts/reports/*.html
       if-no-files-found: warn
   ```

Every CI run therefore produces a downloadable `evidently-reports` artifact —
reviewers can open the actual drift dashboards for the run, not just read a
green checkmark. (`if-no-files-found: warn` keeps the job green if reports were
skipped, consistent with the graceful-degradation design.)

---

## 2. Scheduling drift checks

In production, drift detection is a **recurring job**, not a push-triggered one.
GitHub Actions supports cron directly:

```yaml
# .github/workflows/drift-check.yml (suggested — not in the repo yet)
name: Scheduled drift check

on:
  schedule:
    - cron: "0 6 * * *"        # every day at 06:00 UTC
  workflow_dispatch: {}         # allow manual runs too

jobs:
  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      # In production, replace --scenario with pulling the latest labelled
      # batch into data/current.csv, then use --no-regenerate.
      - run: python scripts/run_pipeline.py --scenario moderate --log-monitoring
      - uses: actions/upload-artifact@v4
        with:
          name: drift-reports-${{ github.run_id }}
          path: |
            artifacts/reports/*.html
            artifacts/metrics/*.json
```

Notes:

- GitHub cron granularity is minutes but scheduling is best-effort; for tight
  SLAs use an external scheduler (Airflow, Dagster, cloud cron) invoking the
  same scripts.
- Pass `--log-monitoring` so scheduled runs build the
  [monitoring history](06-monitoring-dashboard.md) time series. On ephemeral CI
  runners, persist `artifacts/metrics/monitoring_history.json` somewhere durable
  (artifact download + re-upload, a bucket, or a small database) between runs.
- The same schedule shape works with any CI system — the entry points are plain
  scripts (`scripts/detect_drift.py`, `scripts/optimize.py`,
  `scripts/run_pipeline.py`) with no GitHub-specific coupling.

---

## 3. Productionising the loop

This repo is a **reference implementation**: the loop is real, but the data is
synthetic and the "deployment" is a joblib file on disk. Here is the honest map
from repo concept to production component:

| In this repo | In production |
|---|---|
| `data/reference.csv` | Training snapshot / a curated baseline window |
| `data/current.csv` (generated per scenario) | The latest labelled production batch (ground truth has arrived) |
| `artifacts/models/champion.joblib` | A model registry entry (MLflow, SageMaker, Vertex, W&B) with stage = *Production* |
| `save_model(...)` on promotion | Registry stage transition + deployment rollout |
| `monitoring_history.json` | Metrics store / observability backend |
| `--scenario severe` | Reality. You don't get to pick. |

Concrete steps, in rough order of value:

1. **Wire a real prediction service.** Log every request's features and
   prediction; once ground truth arrives, join it back to form the "current"
   batch that `detect_dataset_drift(...)` and `optimize_after_detection(...)`
   consume. The optimizer explicitly requires the `target` column — the loop's
   cadence is bounded by your **label latency**.
2. **Move `champion.joblib` into a model registry.** Promotion
   (`outcome.promoted == True`) becomes a registry stage transition, giving you
   versioning, lineage, and instant rollback. Keep the champion/challenger gate
   exactly where it is — it is your deployment gate.
3. **Alert on `dataset_drift == True`.** The pipeline's summary JSON makes this
   a one-liner in any runner:

   ```bash
   python scripts/detect_drift.py --no-reports | python -c "
   import json, sys
   d = json.load(sys.stdin)
   sys.exit(1 if d['dataset_drift'] else 0)"
   ```

   A non-zero exit fails the CI step, which triggers your normal CI alerting
   (Slack/email/PagerDuty webhook).
4. **Gate deploys on the champion/challenger result.** Run
   `scripts/optimize.py` in the pipeline; only if it prints `promoted: true`
   (and saves the new champion) does the deploy step run. Never auto-deploy an
   ungated retrain — that is precisely the failure mode the gate exists for.

---

## 4. Thresholds as alerting policy

The thresholds in [`config/config.yaml`](../config/config.yaml) *are* your
alerting policy — tune them as such:

| Config key | Operational meaning |
|---|---|
| `drift.psi_threshold` (0.2) | Per-feature "major shift" line; lower it to page earlier |
| `drift.dataset_drift_share` (0.5) | How much of the world must change before the **alarm** fires |
| `optimization.min_drift_share_to_retrain` (0.35) | How much drift triggers **action** (note: action can trigger before the 0.5 alarm when performance drops) |
| `optimization.performance_drop_to_retrain` (0.03) | Your tolerance for silent metric degradation |
| `optimization.min_improvement` (0.0) | Promotion insurance; raise it if eval slices are small/noisy |

A sane layered policy: **warn** when any feature crosses PSI 0.2, **alert** when
`dataset_drift` flips true, **act** (retrain + gate) when the policy in
[docs/05](05-optimization-remediation.md) says so, and **page a human** only when
the challenger fails to beat a degraded champion — that is the signal that
automated remediation has hit its limits (e.g. a data bug rather than drift).

---

## 5. Reproducibility

Every stochastic step is seeded and configured from one file:

- `project.random_seed: 42` flows into data generation (reference uses `seed`,
  current uses `seed + 1`), model training, the adaptation/eval
  `train_test_split`, and retraining — two runs of the same scenario produce the
  same drift shares and metrics.
- All thresholds, model params, and paths live in `config/config.yaml`; both
  `scripts/*` and `run_pipeline(...)` accept `--config` / a `Config` object, so
  you can commit environment-specific configs and diff behaviour changes in
  review.
- Each run writes its full evidence trail: `artifacts/metrics/<scenario>_run.json`
  (complete summary), the monitoring history record, and the Evidently HTML
  reports — CI uploads them, so any past decision can be re-examined.

**Caveats, stated plainly:** this is a reference implementation. There is no
authentication, no distributed scheduling, no feature store, no online serving,
and label arrival is assumed rather than engineered. What it does give you is
the *shape* of a closed-loop drift system — detection you can trust offline, a
policy you can audit, remediation you can test, and a promotion gate you can
build a deployment on — with every seam (`scripts/` entry points, config keys,
JSON records) placed where production infrastructure can take over.

That completes the series. For the component-level view of how the pieces
connect, see [architecture.md](architecture.md); to revisit any layer:
[04 Drift Detection](04-drift-detection.md) ·
[05 Optimization & Remediation](05-optimization-remediation.md) ·
[06 Monitoring & Dashboard](06-monitoring-dashboard.md).
