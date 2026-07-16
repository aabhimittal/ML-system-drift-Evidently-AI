# Post-Detection Optimization: Decide, Remediate, Validate, Promote

> Part 5 of the series — prev: [04 Drift Detection](04-drift-detection.md) · next: [06 Monitoring & Dashboard](06-monitoring-dashboard.md)

**This is the core contribution of the project.** Most drift tutorials stop at
the red "drift detected" banner. In production that banner is only the *alarm* —
someone (or something) still has to fix the model. This layer,
`src/ml_drift/optimization/`, is the **closed-loop response**:

```
Detect → Decide → Remediate → Validate → Promote → Monitor
          └────────── this document ──────────┘
```

It is split into three files, each answering one question:

| File | Question | Key export |
|---|---|---|
| `strategies.py` | *What* should we do? | `decide_strategy(...)` → `RemediationDecision` |
| `retrainer.py` | *How* do we build the fix? | `build_training_frame(...)`, `retrain(...)` → `RetrainResult` |
| `optimizer.py` | *Did it actually work?* | `optimize_after_detection(...)` → `OptimizationOutcome` |

---

## 1. The policy: `strategies.py`

The policy balances **two signals**:

1. **How much of the input drifted** — `DriftResult.drift_share` and the set of
   "unstable" features (`DriftResult.unstable_features(psi_drop_feature_threshold)`,
   i.e. PSI ≥ 0.5 by default).
2. **Whether the model is actually hurting** — `performance_drop =
   champion_baseline − champion_metric_on_current` (positive = degraded).

Four strategies, in increasing order of intervention:

```python
class Strategy(str, Enum):
    NONE = "none"
    FEATURE_STABILIZE = "feature_stabilize"
    REWEIGHT = "reweight"
    RETRAIN = "retrain"
```

### 1.1 The exact decision logic

`decide_strategy(drift, performance_drop, *, forced=None, min_drift_share_to_retrain=0.35, performance_drop_to_retrain=0.03, psi_drop_feature_threshold=0.5)`
evaluates rules in order:

0. **Forced override.** If `forced` is set and not `"auto"`, that strategy is
   returned with reason `"forced by config (strategy=...)"` (an unknown name
   falls back to `RETRAIN`).
1. **Keep the champion.** If `not drift.dataset_drift` **and**
   `performance_drop < performance_drop_to_retrain` → `NONE`
   ("drift within tolerance and performance stable").
2. **Retrain — unless the damage is localized.** If
   `drift.drift_share >= min_drift_share_to_retrain` **or**
   `performance_drop >= performance_drop_to_retrain`:
   - **Exception:** if there *are* unstable features, they number at most
     `max(1, n_features // 4)` (a handful), **and** performance is still fine
     (`performance_drop < performance_drop_to_retrain`) → `FEATURE_STABILIZE`
     ("localized instability ...; stabilise then retrain lean model").
   - Otherwise → `RETRAIN`
     ("broad drift and/or performance degradation beyond threshold").
3. **Everything else** (dataset drift declared, but below the retrain bar and
   performance holding) → `REWEIGHT`
   ("moderate drift; adapt by up-weighting recent data").

### 1.2 Decision table

With the defaults (`min_drift_share_to_retrain=0.35`,
`performance_drop_to_retrain=0.03`, `psi_drop_feature_threshold=0.5`):

| `dataset_drift` | `drift_share ≥ 0.35` | `perf_drop ≥ 0.03` | Unstable features localized (≤ n/4) | Strategy |
|---|---|---|---|---|
| No | — | No | — | **NONE** |
| Any | Yes | No | Yes | **FEATURE_STABILIZE** |
| Any | Yes | No | No | **RETRAIN** |
| Any | Any | Yes | — (perf drop disables the exception) | **RETRAIN** |
| Yes | No | No | — | **REWEIGHT** |

As a flowchart:

```
                 forced != auto? ──yes──► that strategy
                        │no
     no dataset drift AND perf_drop < 0.03? ──yes──► NONE
                        │no
 drift_share ≥ 0.35 OR perf_drop ≥ 0.03? ──no──► REWEIGHT
                        │yes
   few unstable features AND perf still OK? ──yes──► FEATURE_STABILIZE
                        │no
                     RETRAIN
```

The result is a `RemediationDecision` dataclass: `strategy`, `reason`,
`drift_share`, `performance_drop`, `unstable_features` — with `.as_dict()` for
logging. Every decision is therefore *auditable*: you can always answer "why did
the system retrain last Tuesday?".

---

## 2. The mechanics: `retrainer.py`

Each strategy assembles its training frame differently.
`build_training_frame(strategy, reference, current, *, target="target",
rolling_window_size=6000, recent_sample_weight=3.0, unstable_features=None)`
returns `(frame, sample_weight, dropped_features)`:

- **`RETRAIN` — rolling window of recent rows.** Concatenate
  `reference + current` and keep only the **tail** `rolling_window_size` rows —
  biased toward current data (current is appended last), topping up from the
  reference only if current is smaller than the window. No sample weights.
  This is the "the world changed, learn the new world" move — essential under
  concept drift, where the old input→label mapping is simply wrong.
- **`REWEIGHT` — remember, but lean new.** Use *all* of `reference + current`,
  with a sample-weight vector: `1.0` for reference rows,
  `recent_sample_weight` (default `3.0`) for current rows. The model adapts to
  the new regime without forgetting the old one — the right call for moderate
  covariate drift.
- **`FEATURE_STABILIZE` — quarantine the culprits.** Use `reference + current`
  but **drop the unstable (high-PSI) features** (never the target). Returns the
  dropped column names so the outcome can report them. Ideal when one upstream
  pipeline broke a couple of features but the signal elsewhere is intact.
- **`NONE` / unknown** — returns a copy of the reference (callers normally skip
  retraining entirely for `NONE`).

`retrain(strategy, reference, current, *, target, model_type, model_params,
test_size=0.25, seed=42, rolling_window_size, recent_sample_weight,
unstable_features)` funnels the frame into the same `train_model(...)` used for
the baseline (so the challenger is a like-for-like pipeline) and returns:

```python
@dataclass
class RetrainResult:
    bundle: ModelBundle          # the trained challenger
    strategy: Strategy
    n_train_rows: int
    dropped_features: List[str]  # non-empty only for FEATURE_STABILIZE
    notes: str                   # e.g. "reweight retrain (recent x3.0)"
```

---

## 3. The orchestration: `optimizer.py`

`optimize_after_detection(champion, reference, current, drift, *, ...)` runs the
full cycle. The steps — and the two design decisions that make it honest:

1. **Split current into adaptation vs held-out evaluation.**
   `train_test_split(current, test_size=eval_fraction, random_state=seed,
   stratify=current[target])` with `eval_fraction=0.3` by default. The eval
   slice is **never trained on** — this prevents leakage that would otherwise
   let the challenger "win" simply by having seen the test data.
2. **Measure the champion's drop.** Score the champion on the eval slice;
   `performance_drop = champion.metrics[primary_metric] − champion_metric`
   (its validation baseline minus its score on the new regime).
3. **Decide** via `decide_strategy(...)` (Section 1), passing the configured
   thresholds and any forced strategy.
4. **`NONE` short-circuits**: return immediately with the champion as
   `best_bundle`, `challenger_metric=None`, `promoted=False`.
5. **Retrain a challenger** with `retrain(...)` on **reference + adaptation
   slice** (never the eval slice).
6. **Evaluate both on the same eval slice.** For `FEATURE_STABILIZE` the
   challenger's bundle selects its own (reduced) feature list, so passing the
   full `x_eval` is safe.
7. **Champion/challenger gate.** With `champion_challenger=True`:

   ```python
   improvement = challenger_metric - champion_metric
   promoted = improvement >= min_improvement and improvement > 0
   ```

   The `> 0` guard means that even with `min_improvement: 0.0` (the default), a
   tie or a regression keeps the champion. With `champion_challenger=False` the
   retrained model is always adopted.

### 3.1 `OptimizationOutcome`

```python
@dataclass
class OptimizationOutcome:
    decision: RemediationDecision
    champion_metric: float            # on the held-out eval slice
    challenger_metric: Optional[float]  # None when strategy == NONE
    promoted: bool
    primary_metric: str               # e.g. "roc_auc"
    best_bundle: ModelBundle          # challenger if promoted, else champion
    improvement: float
    notes: str
    details: Dict[str, Any]           # performance_drop, n_train_rows,
                                      # dropped_features, eval_rows
```

`.summary()` flattens everything (strategy, reason, metrics rounded to 4
decimals, promoted flag, notes, plus the `details`) into a JSON-ready dict —
this is what the pipeline writes to `artifacts/metrics/<scenario>_run.json` and
what feeds the [monitoring history](06-monitoring-dashboard.md).

---

## 4. Worked example: the severe-drift run

From the README's headline run (`python scripts/run_pipeline.py --scenario severe`):

```
Dataset drift: True (7/9 features, share=0.78)
Remediation:   retrain — broad drift and/or performance degradation beyond threshold
Champion  roc_auc on current : 0.464     ← barely better than chance
Challenger roc_auc on current: 0.746     ← remediation recovers performance
Promoted challenger: True
```

Tracing it through the machinery:

- Detection: 7 of 9 features drifted → `drift_share = 0.78 ≥ 0.5` →
  `dataset_drift = True`.
- The severe scenario includes **concept drift**, so the champion's ROC-AUC on
  the eval slice collapses to **0.464**; against its validation baseline this is
  a `performance_drop` far above `0.03`.
- Policy: `drift_share 0.78 ≥ 0.35` *and* a large performance drop → the
  localized exception can't apply → **RETRAIN**.
- Mechanics: rolling window of the most recent 6,000 rows (dominated by the
  adaptation slice of the current batch).
- Validation: challenger scores **0.746** on the held-out slice →
  `improvement = +0.282 ≥ min_improvement (0.0)` and `> 0` → **promoted**, and
  the pipeline overwrites `artifacts/models/champion.joblib`.

---

## 5. Running it

```bash
python scripts/optimize.py --strategy auto        # policy decides (or: make optimize)
python scripts/optimize.py --strategy retrain     # force a full retrain
python scripts/optimize.py --strategy reweight    # force recency re-weighting
python scripts/optimize.py --strategy feature_stabilize
python scripts/optimize.py --strategy none        # measure, but never remediate
```

The script loads the data and `artifacts/models/champion.joblib`, re-runs
detection, calls `optimize_after_detection(...)` with every knob from config,
prints `outcome.summary()` as JSON, and — **only if promoted** — saves the
challenger over `champion.joblib`. Omitting `--strategy` uses
`optimization.strategy` from config.

---

## 6. Configuration reference (`optimization:` block)

From [`config/config.yaml`](../config/config.yaml):

| Key | Default | Meaning |
|---|---|---|
| `strategy` | `auto` | `auto` lets `decide_strategy` pick; any other value forces it |
| `min_drift_share_to_retrain` | `0.35` | drift share at/above which the policy escalates to retrain |
| `performance_drop_to_retrain` | `0.03` | absolute drop in the primary metric that triggers retraining |
| `retrain_window` | `rolling` | documented window mode (the retrainer implements the rolling tail) |
| `rolling_window_size` | `6000` | rows kept for the `retrain` strategy's window |
| `recent_sample_weight` | `3.0` | up-weight factor for current rows under `reweight` |
| `psi_drop_feature_threshold` | `0.5` | PSI at/above which a feature counts as "unstable" |
| `champion_challenger` | `true` | gate promotion on beating the champion (set `false` to always adopt) |
| `min_improvement` | `0.0` | required metric gain to promote (the gate also requires `> 0`) |
| `max_retrain_rounds` | `3` | budget for repeated remediation rounds |

Tuning intuition: `performance_drop_to_retrain` is your tolerance for silent
degradation; `min_improvement` is your insurance against noisy promotions (raise
it, e.g. to `0.01`, if eval slices are small); `psi_drop_feature_threshold`
controls how eagerly the system quarantines features instead of retraining.

Next: every one of these runs gets logged and charted →
[06 Monitoring & Dashboard](06-monitoring-dashboard.md).
