"""Drift detection: a native statistical core plus Evidently AI reports."""

from .detector import (
    DriftResult,
    FeatureDrift,
    detect_dataset_drift,
    population_stability_index,
)
from .reports import (
    EVIDENTLY_AVAILABLE,
    build_data_drift_report,
    build_performance_report,
    build_target_drift_report,
    generate_all_reports,
)

__all__ = [
    "DriftResult",
    "FeatureDrift",
    "detect_dataset_drift",
    "population_stability_index",
    "EVIDENTLY_AVAILABLE",
    "build_data_drift_report",
    "build_target_drift_report",
    "build_performance_report",
    "generate_all_reports",
]
