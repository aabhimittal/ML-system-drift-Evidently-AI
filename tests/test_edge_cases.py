"""Industrial edge cases for the drift detector — the stuff that breaks in prod.

Each test feeds the detector a pathological batch and asserts it returns a sane
result instead of raising. These are the failure modes real pipelines hit:
schema mismatch, single-class/empty batches, constant or all-NaN columns,
unseen categories, mixed dtypes, infinities, and extreme/degenerate drift.
"""
import numpy as np
import pandas as pd
import pytest

from ml_drift.drift.detector import detect_dataset_drift, population_stability_index


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "num": rng.normal(0, 1, n),
        "cat": rng.choice(["a", "b", "c"], n),
        "target": rng.integers(0, 2, n),
    })


def test_missing_column_in_current_is_ignored():
    ref = _frame()
    cur = _frame(seed=1).drop(columns=["num"])
    result = detect_dataset_drift(ref, cur)
    # Only 'cat' is common; detector must not crash and must not test 'num'.
    assert all(f.feature != "num" for f in result.features)


def test_extra_column_in_current_is_ignored():
    ref = _frame()
    cur = _frame(seed=1)
    cur["extra"] = 1.0
    result = detect_dataset_drift(ref, cur)
    assert all(f.feature != "extra" for f in result.features)


def test_empty_current_batch():
    ref = _frame()
    cur = ref.iloc[0:0].copy()
    result = detect_dataset_drift(ref, cur)
    assert result.dataset_drift is False
    assert result.n_features >= 1


def test_single_row_current_batch():
    ref = _frame()
    cur = _frame(seed=5).iloc[:1].copy()
    result = detect_dataset_drift(ref, cur)   # must not raise
    assert isinstance(result.drift_share, float)


def test_all_nan_numeric_column():
    ref = _frame()
    cur = _frame(seed=1)
    cur["num"] = np.nan
    result = detect_dataset_drift(ref, cur)
    num = next(f for f in result.features if f.feature == "num")
    assert num.drifted is False   # nothing to compare -> no false alarm


def test_constant_reference_column():
    ref = _frame()
    ref["num"] = 3.0              # constant in reference
    cur = _frame(seed=1)          # varies in current
    result = detect_dataset_drift(ref, cur)
    num = next(f for f in result.features if f.feature == "num")
    assert num.test_name == "constant-ref"
    assert num.drifted is True    # moved away from the constant


def test_unseen_category_does_not_crash():
    ref = _frame()
    cur = _frame(seed=1)
    cur.loc[cur.index[:50], "cat"] = "z"   # new level
    result = detect_dataset_drift(ref, cur)
    cat = next(f for f in result.features if f.feature == "cat")
    assert cat.drifted in (True, False)     # returns a verdict, no exception


def test_numeric_column_arrives_as_string():
    ref = _frame()
    cur = _frame(seed=1)
    cur["num"] = cur["num"].astype(str)     # schema drift: numeric -> string
    result = detect_dataset_drift(ref, cur)  # coerced internally, no crash
    assert any(f.feature == "num" for f in result.features)


def test_infinities_are_handled():
    ref = _frame()
    cur = _frame(seed=1)
    cur.loc[cur.index[:10], "num"] = np.inf
    cur.loc[cur.index[10:20], "num"] = -np.inf
    result = detect_dataset_drift(ref, cur)   # infinities dropped, no crash
    num = next(f for f in result.features if f.feature == "num")
    assert np.isfinite(num.psi)


def test_single_class_target_column_not_tested_by_default():
    # target is excluded from feature drift by default; ensure that holds.
    ref = _frame()
    cur = _frame(seed=1)
    cur["target"] = 0
    result = detect_dataset_drift(ref, cur)
    assert all(f.feature != "target" for f in result.features)


def test_identical_frames_no_drift():
    ref = _frame()
    result = detect_dataset_drift(ref, ref.copy())
    assert result.dataset_drift is False
    assert result.drift_share == 0.0


def test_extreme_disjoint_drift():
    ref = _frame()
    cur = _frame(seed=1)
    cur["num"] = cur["num"] + 100      # completely disjoint
    result = detect_dataset_drift(ref, cur)
    num = next(f for f in result.features if f.feature == "num")
    assert num.drifted is True
    assert num.js_divergence > 0.5


def test_bh_correction_reduces_false_alarms():
    # Two i.i.d. draws (no true drift): BH should flag no more than raw p-values.
    ref = _frame(n=800, seed=0)
    cur = _frame(n=800, seed=1)
    raw = detect_dataset_drift(ref, cur, correction="none")
    bh = detect_dataset_drift(ref, cur, correction="bh")
    assert bh.n_drifted <= raw.n_drifted


def test_psi_handles_constant():
    assert population_stability_index(np.ones(100), np.ones(100)) == 0.0
