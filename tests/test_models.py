"""Tests for model training, evaluation, persistence and prediction."""
import numpy as np

from ml_drift.models.predict import predict, predict_proba, score_frame
from ml_drift.models.train import evaluate, load_model, save_model, train_model


def test_train_produces_metrics(champion):
    assert 0.0 <= champion.metrics["roc_auc"] <= 1.0
    # The reference relationship is learnable; expect clearly better than chance.
    assert champion.metrics["roc_auc"] > 0.7


def test_feature_partition(champion):
    assert "amount" in champion.numeric_features
    assert "region" in champion.categorical_features
    assert champion.target == "target"


def test_save_and_load_roundtrip(champion, tmp_path, reference_df):
    path = save_model(champion, tmp_path / "m.joblib")
    loaded = load_model(path)
    x = reference_df.drop(columns=["target"])
    assert np.array_equal(predict(champion, x), predict(loaded, x))


def test_score_frame_columns(champion, current_moderate_df):
    scored = score_frame(champion, current_moderate_df)
    assert "prediction" in scored.columns
    assert "prediction_proba" in scored.columns
    assert scored["prediction_proba"].between(0, 1).all()


def test_sample_weight_training(reference_df):
    w = np.ones(len(reference_df))
    bundle = train_model(reference_df, sample_weight=w,
                         params={"n_estimators": 40, "max_depth": 6})
    assert "roc_auc" in bundle.metrics
