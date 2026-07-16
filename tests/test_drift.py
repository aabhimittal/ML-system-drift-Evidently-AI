"""Tests for the native drift-detection core."""
import numpy as np

from ml_drift.drift.detector import (
    detect_dataset_drift,
    population_stability_index,
)


def test_psi_zero_for_identical():
    x = np.random.default_rng(0).normal(size=5000)
    assert population_stability_index(x, x) < 1e-6


def test_psi_positive_for_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)
    assert population_stability_index(ref, cur) > 0.2


def test_no_drift_detected_when_identical_distribution(reference_df, current_none_df):
    result = detect_dataset_drift(reference_df, current_none_df)
    assert result.dataset_drift is False
    assert result.drift_share < 0.5


def test_drift_detected_for_severe(reference_df, current_severe_df):
    result = detect_dataset_drift(reference_df, current_severe_df)
    assert result.dataset_drift is True
    assert "amount" in result.drifted_features
    assert result.n_drifted >= 1


def test_result_serialisation(reference_df, current_moderate_df):
    result = detect_dataset_drift(reference_df, current_moderate_df)
    d = result.as_dict()
    assert set(["dataset_drift", "drift_share", "features"]).issubset(d.keys())
    assert len(d["features"]) == result.n_features


def test_unstable_features_subset(reference_df, current_severe_df):
    result = detect_dataset_drift(reference_df, current_severe_df)
    unstable = result.unstable_features(psi_threshold=0.5)
    assert set(unstable).issubset(set(result.drifted_features))
