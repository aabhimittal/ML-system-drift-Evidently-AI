"""Post-detection optimization / remediation.

Once drift is *detected*, this package decides what to do about it and does it:

* :mod:`strategies`  — policy: pick a remediation strategy from drift + metrics.
* :mod:`retrainer`   — mechanics: rebuild the training set and retrain.
* :mod:`optimizer`   — orchestration: apply the strategy, validate with a
  champion/challenger gate, and report the outcome.
"""

from .strategies import (
    RemediationDecision,
    Strategy,
    decide_strategy,
)
from .retrainer import RetrainResult, build_training_frame, retrain
from .policy import (
    PromotionVerdict,
    expected_value_of_retraining,
    should_promote,
)
from .optimizer import OptimizationOutcome, optimize_after_detection

__all__ = [
    "Strategy",
    "RemediationDecision",
    "decide_strategy",
    "RetrainResult",
    "build_training_frame",
    "retrain",
    "PromotionVerdict",
    "expected_value_of_retraining",
    "should_promote",
    "OptimizationOutcome",
    "optimize_after_detection",
]
