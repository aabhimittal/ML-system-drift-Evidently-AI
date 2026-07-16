"""Retraining mechanics for the remediation strategies.

Each strategy assembles its training frame differently:

* ``RETRAIN``            — rolling window of the most recent (current) labelled data.
* ``REWEIGHT``           — reference + current combined, with current rows
  up-weighted so the model leans toward the new regime without forgetting.
* ``FEATURE_STABILIZE``  — reference + current with the unstable features dropped.

All paths funnel through :func:`retrain`, which returns a fresh
:class:`~ml_drift.models.train.ModelBundle` plus the sample weights used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models.train import ModelBundle, train_model
from .strategies import Strategy


@dataclass
class RetrainResult:
    bundle: ModelBundle
    strategy: Strategy
    n_train_rows: int
    dropped_features: List[str] = field(default_factory=list)
    notes: str = ""


def build_training_frame(
    strategy: Strategy,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    target: str = "target",
    rolling_window_size: int = 6000,
    recent_sample_weight: float = 3.0,
    unstable_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[str]]:
    """Assemble ``(frame, sample_weight, dropped_features)`` for a strategy.

    ``current`` must include the (freshly labelled) ``target`` column — in
    production this is the data whose ground truth has since arrived.
    """
    unstable_features = unstable_features or []

    if strategy == Strategy.RETRAIN:
        # Rolling window: keep only the most recent rows across ref+current,
        # biased to current. Simplest robust choice: take current, then top up
        # from reference if current is smaller than the window.
        combined = pd.concat([reference, current], ignore_index=True)
        frame = combined.tail(rolling_window_size).reset_index(drop=True)
        return frame, None, []

    if strategy == Strategy.REWEIGHT:
        frame = pd.concat([reference, current], ignore_index=True).reset_index(drop=True)
        weights = np.concatenate([
            np.ones(len(reference)),
            np.full(len(current), float(recent_sample_weight)),
        ])
        return frame, weights, []

    if strategy == Strategy.FEATURE_STABILIZE:
        frame = pd.concat([reference, current], ignore_index=True).reset_index(drop=True)
        drop = [c for c in unstable_features if c in frame.columns and c != target]
        frame = frame.drop(columns=drop)
        return frame, None, drop

    # Strategy.NONE or unknown -> just use reference (caller usually skips this).
    return reference.copy(), None, []


def retrain(
    strategy: Strategy,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    target: str = "target",
    model_type: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.25,
    seed: int = 42,
    rolling_window_size: int = 6000,
    recent_sample_weight: float = 3.0,
    unstable_features: Optional[List[str]] = None,
) -> RetrainResult:
    """Retrain a challenger model according to ``strategy``."""
    frame, weights, dropped = build_training_frame(
        strategy,
        reference,
        current,
        target=target,
        rolling_window_size=rolling_window_size,
        recent_sample_weight=recent_sample_weight,
        unstable_features=unstable_features,
    )
    bundle = train_model(
        frame,
        target=target,
        model_type=model_type,
        params=model_params,
        test_size=test_size,
        seed=seed,
        sample_weight=weights,
    )
    notes = {
        Strategy.RETRAIN: "rolling-window retrain on recent data",
        Strategy.REWEIGHT: f"reweight retrain (recent x{recent_sample_weight})",
        Strategy.FEATURE_STABILIZE: f"stabilized retrain, dropped {dropped}",
    }.get(strategy, "retrain")
    return RetrainResult(
        bundle=bundle,
        strategy=strategy,
        n_train_rows=len(frame),
        dropped_features=dropped,
        notes=notes,
    )
