"""Tests for the schema / data-contract validation."""
import numpy as np
import pandas as pd

from ml_drift.drift.schema import SchemaContract, validate_batch


def _base(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "amount": rng.normal(50, 10, n),
        "region": rng.choice(["north", "south", "east"], n),
        "flag": rng.integers(0, 2, n),
    })


def test_clean_batch_passes():
    ref = _base()
    report = validate_batch(ref, _base(seed=1))
    assert report.ok is True
    assert report.n_errors == 0


def test_missing_column_is_error():
    ref = _base()
    batch = _base(seed=1).drop(columns=["amount"])
    report = validate_batch(ref, batch)
    assert report.ok is False
    kinds = {v.kind for v in report.errors}
    assert "missing_column" in kinds


def test_unexpected_column_is_warning():
    ref = _base()
    batch = _base(seed=1)
    batch["surprise"] = 1.0
    report = validate_batch(ref, batch)
    assert report.ok is True                       # warning, not error
    assert any(v.kind == "unexpected_column" for v in report.violations)


def test_dtype_change_numeric_to_string():
    ref = _base()
    batch = _base(seed=1)
    batch["amount"] = batch["amount"].astype(str) + "x"   # unparseable
    report = validate_batch(ref, batch)
    assert any(v.kind == "dtype_change" for v in report.violations)
    assert report.ok is False                       # >50% unparseable => error


def test_unseen_category_is_warning():
    ref = _base()
    batch = _base(seed=1)
    batch.loc[batch.index[:20], "region"] = "westworld"   # unseen level
    report = validate_batch(ref, batch)
    assert any(v.kind == "unseen_category" for v in report.violations)


def test_out_of_range_error_when_widespread():
    ref = _base()
    batch = _base(seed=1)
    batch["amount"] = batch["amount"] + 10_000        # everything out of range
    report = validate_batch(ref, batch)
    oob = [v for v in report.violations if v.kind == "out_of_range"]
    assert oob and oob[0].severity == "error"


def test_null_spike_warning():
    ref = _base()
    batch = _base(seed=1)
    batch.loc[batch.index[:100], "amount"] = np.nan   # 20% nulls
    report = validate_batch(ref, batch)
    assert any(v.kind == "null_spike" for v in report.violations)


def test_contract_infer_and_serialise():
    ref = _base()
    contract = SchemaContract.infer(ref)
    assert "amount" in contract.columns
    assert contract.columns["amount"].is_numeric
    assert contract.columns["region"].allowed_categories is not None
    # Report serialises to a plain dict.
    report = contract.validate(_base(seed=2))
    d = report.as_dict()
    assert set(["ok", "n_errors", "n_warnings", "violations"]).issubset(d.keys())
