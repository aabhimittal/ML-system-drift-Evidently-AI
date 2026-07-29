"""Drift detection: a native statistical core plus Evidently AI reports."""

from .detector import (
    DriftResult,
    FeatureDrift,
    detect_dataset_drift,
    population_stability_index,
)
from .stats import (
    benjamini_hochberg,
    jensen_shannon_divergence,
    normalized_wasserstein,
)
from .advanced import (
    ImpactWeightedDrift,
    PredictionDrift,
    SegmentedDrift,
    detect_prediction_drift,
    impact_weighted_drift,
    model_feature_importances,
    segmented_drift,
)
from .schema import (
    ColumnContract,
    SchemaContract,
    SchemaReport,
    SchemaViolation,
    validate_batch,
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
    "benjamini_hochberg",
    "jensen_shannon_divergence",
    "normalized_wasserstein",
    "ImpactWeightedDrift",
    "PredictionDrift",
    "SegmentedDrift",
    "detect_prediction_drift",
    "impact_weighted_drift",
    "model_feature_importances",
    "segmented_drift",
    "ColumnContract",
    "SchemaContract",
    "SchemaReport",
    "SchemaViolation",
    "validate_batch",
    "EVIDENTLY_AVAILABLE",
    "build_data_drift_report",
    "build_target_drift_report",
    "build_performance_report",
    "generate_all_reports",
]
