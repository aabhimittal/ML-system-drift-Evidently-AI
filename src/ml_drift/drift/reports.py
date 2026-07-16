"""Evidently AI report generation.

These functions produce rich, shareable HTML reports using Evidently's classic
``Report`` + ``metric_preset`` API (evidently >= 0.4, < 0.5). Evidently is an
*optional* dependency: if it is not installed the functions degrade gracefully,
returning ``None`` and leaving the native detector (``detector.py``) as the
source of truth. This keeps the pipeline runnable in minimal environments.

Reports produced
----------------
* **Data drift**   — per-feature distribution drift (``DataDriftPreset``).
* **Target drift** — shift in the target / prediction distribution.
* **Performance**  — classification quality on the current batch.
* **Data quality** — missing values, ranges, cardinality.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

try:  # pragma: no cover - exercised only when evidently is installed
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import (
        ClassificationPreset,
        DataDriftPreset,
        DataQualityPreset,
        TargetDriftPreset,
    )

    EVIDENTLY_AVAILABLE = True
except Exception:  # ImportError or API drift on newer versions
    EVIDENTLY_AVAILABLE = False


def _column_mapping(
    target: Optional[str],
    prediction: Optional[str],
    numeric: List[str],
    categorical: List[str],
):
    mapping = ColumnMapping()
    mapping.target = target
    mapping.prediction = prediction
    mapping.numerical_features = numeric
    mapping.categorical_features = categorical
    return mapping


def _save(report: "Report", out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    return out_path


def build_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_path: str | Path,
    numeric: Optional[List[str]] = None,
    categorical: Optional[List[str]] = None,
) -> Optional[Path]:
    """Generate the per-feature data-drift report."""
    if not EVIDENTLY_AVAILABLE:
        return None
    mapping = _column_mapping(None, None, numeric or [], categorical or [])
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)
    return _save(report, out_path)


def build_target_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_path: str | Path,
    target: str = "target",
    prediction: Optional[str] = "prediction",
) -> Optional[Path]:
    """Generate the target/prediction drift report."""
    if not EVIDENTLY_AVAILABLE:
        return None
    pred = prediction if prediction in current.columns else None
    mapping = _column_mapping(target, pred, [], [])
    report = Report(metrics=[TargetDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)
    return _save(report, out_path)


def build_performance_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_path: str | Path,
    target: str = "target",
    prediction: str = "prediction",
) -> Optional[Path]:
    """Generate the classification-performance report for the current batch."""
    if not EVIDENTLY_AVAILABLE:
        return None
    if prediction not in current.columns or prediction not in reference.columns:
        return None
    mapping = _column_mapping(target, prediction, [], [])
    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)
    return _save(report, out_path)


def build_data_quality_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_path: str | Path,
) -> Optional[Path]:
    """Generate the data-quality report."""
    if not EVIDENTLY_AVAILABLE:
        return None
    report = Report(metrics=[DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    return _save(report, out_path)


def generate_all_reports(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reports_dir: str | Path,
    target: str = "target",
    prediction: str = "prediction",
    numeric: Optional[List[str]] = None,
    categorical: Optional[List[str]] = None,
    prefix: str = "",
) -> dict:
    """Generate the full report suite; returns a mapping of name -> path (or None).

    Never raises if Evidently is missing — callers can rely on the native
    detector for the drift decision and treat reports as a nice-to-have.
    """
    reports_dir = Path(reports_dir)
    p = f"{prefix}_" if prefix else ""
    results = {
        "data_drift": build_data_drift_report(
            reference, current, reports_dir / f"{p}data_drift.html", numeric, categorical
        ),
        "target_drift": build_target_drift_report(
            reference, current, reports_dir / f"{p}target_drift.html", target, prediction
        ),
        "performance": build_performance_report(
            reference, current, reports_dir / f"{p}performance.html", target, prediction
        ),
        "data_quality": build_data_quality_report(
            reference, current, reports_dir / f"{p}data_quality.html"
        ),
    }
    return {k: (str(v) if v else None) for k, v in results.items()}
