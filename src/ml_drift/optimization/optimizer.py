"""Orchestration of post-detection optimization.

``optimize_after_detection`` is the entry point the pipeline calls once drift
has been detected. It:

1. Splits the freshly-labelled ``current`` batch into an *adaptation* slice
   (used for retraining) and a held-out *evaluation* slice (never trained on),
   so champion and challenger are compared fairly on the new regime.
2. Measures the champion's performance drop on the evaluation slice.
3. Picks a remediation strategy (:func:`decide_strategy`).
4. Retrains a challenger with that strategy on ``reference + adaptation``.
5. Applies a **champion/challenger gate**: the challenger is only promoted if it
   beats the champion on the evaluation slice by ``min_improvement``.

The returned :class:`OptimizationOutcome` records every number so the decision
is auditable and can be logged to the monitoring history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from ..drift.detector import DriftResult
from ..models.train import ModelBundle, evaluate
from .policy import should_promote
from .retrainer import retrain
from .strategies import RemediationDecision, Strategy, decide_strategy


@dataclass
class OptimizationOutcome:
    decision: RemediationDecision
    champion_metric: float
    challenger_metric: Optional[float]
    promoted: bool
    primary_metric: str
    best_bundle: ModelBundle
    improvement: float = 0.0
    notes: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "strategy": self.decision.strategy.value,
            "reason": self.decision.reason,
            "primary_metric": self.primary_metric,
            "champion_metric": round(self.champion_metric, 4),
            "challenger_metric": (
                round(self.challenger_metric, 4)
                if self.challenger_metric is not None
                else None
            ),
            "improvement": round(self.improvement, 4),
            "promoted": self.promoted,
            "notes": self.notes,
            **self.details,
        }


def _metric(bundle: ModelBundle, x: pd.DataFrame, y: pd.Series, key: str) -> float:
    return evaluate(bundle, x, y).get(key, float("nan"))


def optimize_after_detection(
    champion: ModelBundle,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift: DriftResult,
    *,
    target: str = "target",
    primary_metric: str = "roc_auc",
    strategy: str = "auto",
    model_type: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
    min_drift_share_to_retrain: float = 0.35,
    performance_drop_to_retrain: float = 0.03,
    psi_drop_feature_threshold: float = 0.5,
    rolling_window_size: int = 6000,
    recent_sample_weight: float = 3.0,
    champion_challenger: bool = True,
    min_improvement: float = 0.0,
    eval_fraction: float = 0.3,
    seed: int = 42,
    value_per_metric_point: Optional[float] = None,
    retrain_cost: float = 0.0,
) -> OptimizationOutcome:
    """Run the full remediation cycle and return the outcome.

    ``current`` must contain the ``target`` column (freshly-arrived ground truth).

    If ``value_per_metric_point`` is provided, an **economic gate** is applied on
    top of the quality gate: the challenger is promoted only if the expected value
    of its improvement exceeds ``retrain_cost`` (see :mod:`optimization.policy`).
    """
    # 1. Hold out an evaluation slice of the current data (never trained on).
    stratify = current[target] if current[target].nunique() > 1 else None
    adapt_df, eval_df = train_test_split(
        current, test_size=eval_fraction, random_state=seed, stratify=stratify
    )
    x_eval = eval_df.drop(columns=[target])
    y_eval = eval_df[target]

    # 2. Champion performance on the new regime.
    champion_metric = _metric(champion, x_eval, y_eval, primary_metric)
    champion_baseline = champion.metrics.get(primary_metric, champion_metric)
    performance_drop = float(champion_baseline - champion_metric)

    # 3. Decide the remediation strategy.
    decision = decide_strategy(
        drift,
        performance_drop,
        forced=strategy,
        min_drift_share_to_retrain=min_drift_share_to_retrain,
        performance_drop_to_retrain=performance_drop_to_retrain,
        psi_drop_feature_threshold=psi_drop_feature_threshold,
    )

    # 4. If policy says do nothing, keep the champion.
    if decision.strategy == Strategy.NONE:
        return OptimizationOutcome(
            decision=decision,
            champion_metric=champion_metric,
            challenger_metric=None,
            promoted=False,
            primary_metric=primary_metric,
            best_bundle=champion,
            improvement=0.0,
            notes="no remediation required; champion retained",
            details={"performance_drop": round(performance_drop, 4)},
        )

    # 5. Retrain a challenger using the chosen strategy (train on ref + adapt).
    retrain_result = retrain(
        decision.strategy,
        reference,
        adapt_df,
        target=target,
        model_type=model_type,
        model_params=model_params,
        seed=seed,
        rolling_window_size=rolling_window_size,
        recent_sample_weight=recent_sample_weight,
        unstable_features=decision.unstable_features,
    )
    challenger = retrain_result.bundle

    # 6. Evaluate the challenger on the same held-out slice.
    #    For FEATURE_STABILIZE the challenger drops columns; evaluate() selects by
    #    the bundle's own feature list, so passing the full x_eval is safe.
    challenger_metric = _metric(challenger, x_eval, y_eval, primary_metric)
    improvement = float(challenger_metric - champion_metric)

    # 7. Champion/challenger gate (quality + optional economic gate).
    verdict = should_promote(
        champion_metric,
        challenger_metric,
        min_improvement=min_improvement,
        champion_challenger=champion_challenger,
        value_per_metric_point=value_per_metric_point,
        retrain_cost=retrain_cost,
    )
    promoted = verdict.promote

    best = challenger if promoted else champion
    notes = f"{retrain_result.notes}; {verdict.reason}"
    details = {
        "performance_drop": round(performance_drop, 4),
        "n_train_rows": retrain_result.n_train_rows,
        "dropped_features": retrain_result.dropped_features,
        "eval_rows": int(len(eval_df)),
    }
    if verdict.expected_value is not None:
        details["expected_value"] = round(verdict.expected_value, 4)
    return OptimizationOutcome(
        decision=decision,
        champion_metric=champion_metric,
        challenger_metric=challenger_metric,
        promoted=promoted,
        primary_metric=primary_metric,
        best_bundle=best,
        improvement=improvement,
        notes=notes,
        details=details,
    )
