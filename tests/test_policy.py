"""Tests for the cost/value-aware remediation gate."""
from ml_drift.optimization.policy import (
    expected_value_of_retraining,
    should_promote,
)


def test_expected_value_positive_and_negative():
    # +0.05 metric gain worth $1000/point => $50 value, $10 cost => +$40.
    assert expected_value_of_retraining(0.05, 1000, 10) == 40.0
    # Tiny gain, big cost => negative EV.
    assert expected_value_of_retraining(0.001, 1000, 10) < 0


def test_quality_gate_rejects_no_improvement():
    v = should_promote(0.80, 0.80, min_improvement=0.0)
    assert v.promote is False


def test_quality_gate_accepts_improvement():
    v = should_promote(0.80, 0.85, min_improvement=0.0)
    assert v.promote is True


def test_min_improvement_threshold():
    v = should_promote(0.80, 0.805, min_improvement=0.02)
    assert v.promote is False        # 0.005 gain below 0.02 bar


def test_disabled_gate_always_promotes():
    v = should_promote(0.90, 0.10, champion_challenger=False)
    assert v.promote is True


def test_economic_gate_blocks_expensive_marginal_gain():
    # Real but tiny improvement whose value doesn't cover the retrain cost.
    v = should_promote(0.80, 0.801, min_improvement=0.0,
                       value_per_metric_point=100, retrain_cost=50)
    assert v.promote is False
    assert v.expected_value is not None and v.expected_value <= 0


def test_economic_gate_allows_worthwhile_gain():
    v = should_promote(0.80, 0.90, min_improvement=0.0,
                       value_per_metric_point=1000, retrain_cost=50)
    assert v.promote is True
    assert v.expected_value > 0


def test_verdict_serialises():
    v = should_promote(0.80, 0.90, value_per_metric_point=1000, retrain_cost=10)
    d = v.as_dict()
    assert set(["promote", "reason", "expected_value"]).issubset(d.keys())
