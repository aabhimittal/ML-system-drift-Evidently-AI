"""Prediction helpers built on top of a trained :class:`ModelBundle`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # avoid importing at runtime to keep this module light
    from .train import ModelBundle


def _features(bundle: "ModelBundle", frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in bundle.feature_names if c in frame.columns]
    return frame[cols]


def predict(bundle: "ModelBundle", frame: pd.DataFrame) -> np.ndarray:
    """Return hard class predictions for ``frame``."""
    return bundle.pipeline.predict(_features(bundle, frame))


def predict_proba(bundle: "ModelBundle", frame: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities for ``frame``."""
    return bundle.pipeline.predict_proba(_features(bundle, frame))[:, 1]


def score_frame(bundle: "ModelBundle", frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ``prediction`` and ``prediction_proba`` columns to a copy of ``frame``.

    This is exactly the shape Evidently expects for its classification and
    target-drift reports (a column of predictions alongside the target).
    """
    out = frame.copy()
    out["prediction"] = predict(bundle, frame)
    out["prediction_proba"] = predict_proba(bundle, frame)
    return out
