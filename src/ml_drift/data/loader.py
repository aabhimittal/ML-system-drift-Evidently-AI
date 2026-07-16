"""Small I/O helpers for reading/writing dataset frames and splitting X/y."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def save_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Generate it first (make data)."
        )
    return pd.read_csv(path)


def split_features_target(
    frame: pd.DataFrame, target: str = "target"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a frame into feature matrix ``X`` and target vector ``y``."""
    if target not in frame.columns:
        raise KeyError(f"Target column '{target}' not present in frame.")
    x = frame.drop(columns=[target])
    y = frame[target]
    return x, y


def feature_columns(frame: pd.DataFrame, target: str = "target") -> List[str]:
    return [c for c in frame.columns if c != target]
