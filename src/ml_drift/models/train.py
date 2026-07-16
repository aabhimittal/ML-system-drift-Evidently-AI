"""Baseline model: build, train, evaluate and persist.

We wrap a scikit-learn pipeline (preprocessing + estimator) together with the
column layout in a :class:`ModelBundle` so downstream code (prediction, drift
reporting, retraining) can rely on a single serialisable artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelBundle:
    """A trained pipeline plus the metadata needed to reuse it."""

    pipeline: Pipeline
    numeric_features: List[str]
    categorical_features: List[str]
    target: str
    model_type: str
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def feature_names(self) -> List[str]:
        return self.numeric_features + self.categorical_features


def _infer_feature_types(x: pd.DataFrame):
    numeric = x.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    return numeric, categorical


def build_estimator(model_type: str, params: Optional[Dict[str, Any]] = None):
    """Return an untrained estimator for the requested ``model_type``."""
    params = params or {}
    model_type = model_type.lower()
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    if model_type == "gradient_boosting":
        # GradientBoosting does not accept n_jobs / some RF params; filter.
        allowed = {k: v for k, v in params.items() if k in {
            "n_estimators", "max_depth", "learning_rate", "subsample",
        }}
        return GradientBoostingClassifier(random_state=42, **allowed)
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    raise ValueError(f"Unknown model_type: {model_type}")


def _build_pipeline(
    numeric: List[str], categorical: List[str], model_type: str, params: Dict[str, Any]
) -> Pipeline:
    # OneHotEncoder API changed across sklearn versions (sparse -> sparse_output).
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", ohe, categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", build_estimator(model_type, params))])


def evaluate(bundle: ModelBundle, x: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Compute a standard classification metric set on ``(x, y)``."""
    pipe = bundle.pipeline
    preds = pipe.predict(x)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }
    # ROC-AUC needs probabilities and both classes present.
    if hasattr(pipe, "predict_proba") and len(np.unique(y)) > 1:
        proba = pipe.predict_proba(x)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
    else:  # pragma: no cover - degenerate case
        metrics["roc_auc"] = float("nan")
    return metrics


def train_model(
    frame: pd.DataFrame,
    target: str = "target",
    model_type: str = "random_forest",
    params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.25,
    seed: int = 42,
    sample_weight: Optional[np.ndarray] = None,
) -> ModelBundle:
    """Train a model on ``frame`` and return a :class:`ModelBundle`.

    ``sample_weight`` (optional) lets the reweight remediation strategy give more
    influence to recent/current rows during retraining.
    """
    params = params or {}
    x = frame.drop(columns=[target])
    y = frame[target]
    numeric, categorical = _infer_feature_types(x)

    stratify = y if y.nunique() > 1 else None
    if sample_weight is not None:
        x_tr, x_te, y_tr, y_te, w_tr, _ = train_test_split(
            x, y, sample_weight, test_size=test_size, random_state=seed, stratify=stratify
        )
    else:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=stratify
        )
        w_tr = None

    pipe = _build_pipeline(numeric, categorical, model_type, params)
    if w_tr is not None:
        pipe.fit(x_tr, y_tr, model__sample_weight=w_tr)
    else:
        pipe.fit(x_tr, y_tr)

    bundle = ModelBundle(
        pipeline=pipe,
        numeric_features=numeric,
        categorical_features=categorical,
        target=target,
        model_type=model_type,
    )
    bundle.metrics = evaluate(bundle, x_te, y_te)
    return bundle


def save_model(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(path: str | Path) -> ModelBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Train it first.")
    return joblib.load(path)
