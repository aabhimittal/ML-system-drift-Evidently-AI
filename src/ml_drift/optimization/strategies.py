"""Remediation policy: choose *how* to respond to detected drift.

The decision balances two signals:

1. **How much of the input drifted** — the share of drifted features and the
   set of "unstable" features (very high PSI).
2. **Whether the model is actually hurting** — the drop in the primary metric
   (e.g. ROC-AUC) on the freshly-labelled current batch versus the champion's
   validation score.

Strategies (in increasing order of intervention)
------------------------------------------------
* ``NONE``            — drift is within tolerance; keep the champion.
* ``FEATURE_STABILIZE`` — a few features are wildly unstable but overall
  performance holds: drop/quarantine those features and retrain a leaner model.
* ``REWEIGHT``        — moderate drift with some performance loss: retrain on
  reference+current with recent rows up-weighted so the model adapts.
* ``RETRAIN``         — broad drift and/or a real performance drop: retrain on a
  rolling window of the most recent (current) data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ..drift.detector import DriftResult


class Strategy(str, Enum):
    NONE = "none"
    FEATURE_STABILIZE = "feature_stabilize"
    REWEIGHT = "reweight"
    RETRAIN = "retrain"


@dataclass
class RemediationDecision:
    strategy: Strategy
    reason: str
    drift_share: float
    performance_drop: float
    unstable_features: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy.value,
            "reason": self.reason,
            "drift_share": round(self.drift_share, 4),
            "performance_drop": round(self.performance_drop, 4),
            "unstable_features": self.unstable_features,
        }


def decide_strategy(
    drift: DriftResult,
    performance_drop: float,
    *,
    forced: Optional[str] = None,
    min_drift_share_to_retrain: float = 0.35,
    performance_drop_to_retrain: float = 0.03,
    psi_drop_feature_threshold: float = 0.5,
) -> RemediationDecision:
    """Select a remediation strategy.

    Parameters
    ----------
    drift: result from :func:`detect_dataset_drift`.
    performance_drop: champion_metric - current_metric (positive = degraded).
    forced: if set to a strategy name (other than ``"auto"``), bypass the policy.
    """
    unstable = drift.unstable_features(psi_drop_feature_threshold)

    # Explicit override from config (strategy != auto).
    if forced and forced != "auto":
        try:
            strat = Strategy(forced)
        except ValueError:
            strat = Strategy.RETRAIN
        return RemediationDecision(
            strat, f"forced by config (strategy={forced})",
            drift.drift_share, performance_drop, unstable,
        )

    # 1. No meaningful drift and no performance loss -> keep the champion.
    if not drift.dataset_drift and performance_drop < performance_drop_to_retrain:
        return RemediationDecision(
            Strategy.NONE,
            "drift within tolerance and performance stable",
            drift.drift_share, performance_drop, unstable,
        )

    # 2. Broad drift or a real performance drop -> full retrain on recent data.
    if drift.drift_share >= min_drift_share_to_retrain or performance_drop >= performance_drop_to_retrain:
        # If only a handful of features are the culprits and performance is fine,
        # prefer stabilising those features over a full retrain.
        if (
            unstable
            and len(unstable) <= max(1, drift.n_features // 4)
            and performance_drop < performance_drop_to_retrain
        ):
            return RemediationDecision(
                Strategy.FEATURE_STABILIZE,
                f"localized instability in {unstable}; stabilise then retrain lean model",
                drift.drift_share, performance_drop, unstable,
            )
        return RemediationDecision(
            Strategy.RETRAIN,
            "broad drift and/or performance degradation beyond threshold",
            drift.drift_share, performance_drop, unstable,
        )

    # 3. Some drift but below the retrain bar -> adapt via reweighting.
    return RemediationDecision(
        Strategy.REWEIGHT,
        "moderate drift; adapt by up-weighting recent data",
        drift.drift_share, performance_drop, unstable,
    )
