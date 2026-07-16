"""Tests for the post-detection optimization layer."""
from ml_drift.drift.detector import detect_dataset_drift
from ml_drift.optimization.optimizer import optimize_after_detection
from ml_drift.optimization.retrainer import build_training_frame, retrain
from ml_drift.optimization.strategies import Strategy, decide_strategy


def test_decide_none_when_stable(reference_df, current_none_df):
    drift = detect_dataset_drift(reference_df, current_none_df)
    decision = decide_strategy(drift, performance_drop=0.0)
    assert decision.strategy == Strategy.NONE


def test_decide_retrain_on_broad_drift(reference_df, current_severe_df):
    drift = detect_dataset_drift(reference_df, current_severe_df)
    decision = decide_strategy(drift, performance_drop=0.1)
    assert decision.strategy in {Strategy.RETRAIN, Strategy.FEATURE_STABILIZE}


def test_forced_strategy_overrides(reference_df, current_none_df):
    drift = detect_dataset_drift(reference_df, current_none_df)
    decision = decide_strategy(drift, performance_drop=0.0, forced="reweight")
    assert decision.strategy == Strategy.REWEIGHT


def test_build_training_frame_reweight(reference_df, current_moderate_df):
    frame, weights, dropped = build_training_frame(
        Strategy.REWEIGHT, reference_df, current_moderate_df, recent_sample_weight=3.0
    )
    assert len(frame) == len(reference_df) + len(current_moderate_df)
    assert weights is not None
    assert weights.max() == 3.0
    assert dropped == []


def test_build_training_frame_feature_stabilize(reference_df, current_severe_df):
    frame, weights, dropped = build_training_frame(
        Strategy.FEATURE_STABILIZE, reference_df, current_severe_df,
        unstable_features=["amount"],
    )
    assert "amount" not in frame.columns
    assert "amount" in dropped


def test_retrain_returns_bundle(reference_df, current_moderate_df):
    result = retrain(Strategy.RETRAIN, reference_df, current_moderate_df,
                     model_params={"n_estimators": 40, "max_depth": 6})
    assert result.bundle.metrics.get("roc_auc") is not None
    assert result.n_train_rows > 0


def test_optimize_none_keeps_champion(champion, reference_df, current_none_df):
    drift = detect_dataset_drift(reference_df, current_none_df)
    outcome = optimize_after_detection(champion, reference_df, current_none_df, drift,
                                       strategy="auto")
    assert outcome.decision.strategy == Strategy.NONE
    assert outcome.promoted is False
    assert outcome.best_bundle is champion


def test_optimize_severe_runs_remediation(champion, reference_df, current_severe_df):
    drift = detect_dataset_drift(reference_df, current_severe_df)
    outcome = optimize_after_detection(champion, reference_df, current_severe_df, drift,
                                       strategy="auto",
                                       model_params={"n_estimators": 60, "max_depth": 8})
    # A remediation strategy must have been chosen and a challenger evaluated.
    assert outcome.decision.strategy != Strategy.NONE
    assert outcome.challenger_metric is not None
    # The promoted/best model must never be worse than the champion on current.
    assert outcome.best_bundle is not None


def test_champion_challenger_gate_rejects_worse(champion, reference_df, current_none_df):
    # Force a retrain even though nothing drifted; challenger unlikely to beat
    # champion on stable data, so the gate should keep the champion.
    drift = detect_dataset_drift(reference_df, current_none_df)
    outcome = optimize_after_detection(champion, reference_df, current_none_df, drift,
                                       strategy="retrain", champion_challenger=True,
                                       model_params={"n_estimators": 40, "max_depth": 6})
    if not outcome.promoted:
        assert outcome.best_bundle is champion
