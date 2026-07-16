"""Tests for synthetic data generation and drift injection."""
import numpy as np

from ml_drift.data.generate import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    DriftSpec,
    generate_dataset,
    make_reference_and_current,
)


def test_generate_shape_and_columns():
    df = generate_dataset(500, seed=0)
    assert len(df) == 500
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["target"]:
        assert col in df.columns
    assert set(df["target"].unique()).issubset({0, 1})


def test_reproducible():
    a = generate_dataset(300, seed=7)
    b = generate_dataset(300, seed=7)
    assert a.equals(b)


def test_no_drift_spec_matches_distribution():
    ref = generate_dataset(4000, seed=10, spec=DriftSpec.none())
    cur = generate_dataset(4000, seed=11, spec=DriftSpec.none())
    # Means of numeric features should be close when no drift is injected.
    for col in NUMERIC_FEATURES:
        assert abs(ref[col].mean() - cur[col].mean()) < 0.25 * ref[col].std()


def test_mean_shift_moves_distribution():
    ref = generate_dataset(4000, seed=10, spec=DriftSpec.none())
    cur = generate_dataset(4000, seed=11, spec=DriftSpec(numeric_mean_shift={"amount": 2.0}))
    assert cur["amount"].mean() - ref["amount"].mean() > ref["amount"].std()


def test_make_reference_and_current():
    ref, cur = make_reference_and_current(1000, seed=5)
    assert len(ref) == len(cur) == 1000
    assert "target" in ref.columns and "target" in cur.columns
