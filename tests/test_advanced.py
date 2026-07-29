"""Tests for advanced drift analytics: impact-weighted, prediction, segmented."""
import numpy as np
import pandas as pd

from ml_drift.drift.advanced import (
    detect_prediction_drift,
    impact_weighted_drift,
    model_feature_importances,
    segmented_drift,
)
from ml_drift.drift.detector import detect_dataset_drift
from ml_drift.models.predict import predict_proba


# --- impact-weighted drift ------------------------------------------------- #
def test_model_feature_importances_sum_to_one(champion):
    imp = model_feature_importances(champion)
    assert set(imp.keys()) == set(champion.feature_names)
    assert abs(sum(imp.values()) - 1.0) < 1e-6
    assert all(v >= 0 for v in imp.values())


def test_impact_weighted_ranks_important_drift_first(champion, reference_df, current_severe_df):
    drift = detect_dataset_drift(reference_df, current_severe_df)
    imp = model_feature_importances(champion)
    iwd = impact_weighted_drift(drift, imp)
    # Rankings sorted by impact descending; total_impact positive under severe drift.
    impacts = [r.impact for r in iwd.rankings]
    assert impacts == sorted(impacts, reverse=True)
    assert iwd.total_impact > 0
    assert iwd.top_feature is not None


def test_impact_zero_when_no_importance(reference_df, current_severe_df):
    drift = detect_dataset_drift(reference_df, current_severe_df)
    iwd = impact_weighted_drift(drift, {})   # no importances known
    assert iwd.total_impact == 0.0


# --- unsupervised prediction drift ----------------------------------------- #
def test_prediction_drift_none_when_scores_stable(champion, reference_df, current_none_df):
    ref_scores = predict_proba(champion, reference_df)
    cur_scores = predict_proba(champion, current_none_df)
    pd_result = detect_prediction_drift(ref_scores, cur_scores)
    assert pd_result.drifted is False


def test_prediction_drift_detected_under_severe(champion, reference_df, current_severe_df):
    ref_scores = predict_proba(champion, reference_df)
    cur_scores = predict_proba(champion, current_severe_df)
    pd_result = detect_prediction_drift(ref_scores, cur_scores)
    assert pd_result.drifted is True
    assert 0.0 <= pd_result.js_divergence <= 1.0


def test_prediction_drift_empty_inputs():
    res = detect_prediction_drift(np.array([]), np.array([]))
    assert res.drifted is False


# --- segmented drift ------------------------------------------------------- #
def test_segmented_drift_per_region(reference_df, current_severe_df):
    seg = segmented_drift(reference_df, current_severe_df, "region", min_segment_size=50)
    assert seg.segment_column == "region"
    assert len(seg.segments) >= 1
    worst = seg.worst_segment
    assert worst is not None
    name, share = worst
    assert 0.0 <= share <= 1.0


def test_segmented_drift_missing_column_raises(reference_df, current_none_df):
    import pytest
    with pytest.raises(KeyError):
        segmented_drift(reference_df, current_none_df, "not_a_column")


def test_segmented_drift_skips_small_segments(reference_df, current_none_df):
    # Absurdly large min size => no segment qualifies.
    seg = segmented_drift(reference_df, current_none_df, "region", min_segment_size=10**9)
    assert seg.segments == {}
    assert seg.worst_segment is None
