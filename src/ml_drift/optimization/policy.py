"""Cost/value-aware remediation gating.

A champion/challenger gate that only asks "did the metric improve?" ignores that
**retraining is not free** — compute, review time, deployment risk. In an
industrial setting you retrain when the *expected value* of the improvement
outweighs its cost, not merely when the challenger is a hair better.

:func:`expected_value_of_retraining` turns a metric gain into a business value
and nets off the retraining cost. :func:`should_promote` combines the raw metric
gate with this economic gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromotionVerdict:
    promote: bool
    reason: str
    expected_value: Optional[float] = None

    def as_dict(self):
        return {
            "promote": self.promote,
            "reason": self.reason,
            "expected_value": (round(self.expected_value, 4)
                               if self.expected_value is not None else None),
        }


def expected_value_of_retraining(
    metric_gain: float,
    value_per_metric_point: float,
    retrain_cost: float,
) -> float:
    """Net expected value of a retrain.

    ``metric_gain``           challenger_metric - champion_metric (e.g. +0.05 AUC).
    ``value_per_metric_point`` business value of a full 1.0 gain in the metric
                              (e.g. $ saved if AUC went 0 -> 1); scales the gain.
    ``retrain_cost``          fixed cost of retraining + deploying.

    Returns ``metric_gain * value_per_metric_point - retrain_cost``. Positive =>
    worth doing.
    """
    return metric_gain * value_per_metric_point - retrain_cost


def should_promote(
    champion_metric: float,
    challenger_metric: float,
    *,
    min_improvement: float = 0.0,
    champion_challenger: bool = True,
    value_per_metric_point: Optional[float] = None,
    retrain_cost: float = 0.0,
) -> PromotionVerdict:
    """Decide whether to promote the challenger.

    Two gates, applied in order:

    1. **Quality gate** (always): challenger must beat the champion by at least
       ``min_improvement`` (skipped if ``champion_challenger`` is False, in which
       case the retrained model is always adopted).
    2. **Economic gate** (only if ``value_per_metric_point`` is provided): the
       expected value of the improvement must exceed the retraining cost.
    """
    improvement = challenger_metric - champion_metric

    if not champion_challenger:
        return PromotionVerdict(True, "champion/challenger disabled; adopt retrained model")

    if not (improvement >= min_improvement and improvement > 0):
        return PromotionVerdict(
            False,
            f"quality gate: improvement {improvement:+.4f} below "
            f"min_improvement {min_improvement:+.4f}",
        )

    if value_per_metric_point is not None:
        ev = expected_value_of_retraining(improvement, value_per_metric_point, retrain_cost)
        if ev <= 0:
            return PromotionVerdict(
                False,
                f"economic gate: expected value {ev:+.2f} does not cover "
                f"retrain cost {retrain_cost:.2f}",
                expected_value=ev,
            )
        return PromotionVerdict(
            True,
            f"quality + economic gates passed (EV {ev:+.2f})",
            expected_value=ev,
        )

    return PromotionVerdict(True, f"quality gate passed (improvement {improvement:+.4f})")
