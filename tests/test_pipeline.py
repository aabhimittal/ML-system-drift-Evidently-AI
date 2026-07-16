"""End-to-end pipeline smoke tests using a temp-dir config."""
import yaml

from ml_drift.config import load_config
from ml_drift.pipeline.orchestrator import run_pipeline


def _write_config(tmp_path):
    """Create a small config pointing all artifact paths into tmp_path."""
    cfg = {
        "project": {"name": "test", "random_seed": 42},
        "paths": {
            "data_dir": str(tmp_path / "data"),
            "reference_data": str(tmp_path / "data/reference.csv"),
            "current_data": str(tmp_path / "data/current.csv"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "model_dir": str(tmp_path / "artifacts/models"),
            "reports_dir": str(tmp_path / "artifacts/reports"),
            "metrics_dir": str(tmp_path / "artifacts/metrics"),
        },
        "data": {"n_samples": 1500, "drift_intensity": 0.0,
                 "numeric_features": [], "categorical_features": [], "target": "target"},
        "model": {"type": "random_forest", "test_size": 0.25,
                  "params": {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 5}},
        "drift": {"psi_threshold": 0.2, "ks_pvalue_threshold": 0.05,
                  "chi2_pvalue_threshold": 0.05, "dataset_drift_share": 0.5,
                  "target_drift_threshold": 0.1},
        "optimization": {"strategy": "auto", "min_drift_share_to_retrain": 0.35,
                         "performance_drop_to_retrain": 0.03, "retrain_window": "rolling",
                         "rolling_window_size": 3000, "recent_sample_weight": 3.0,
                         "psi_drop_feature_threshold": 0.5, "champion_challenger": True,
                         "min_improvement": 0.0, "max_retrain_rounds": 3},
        "monitoring": {"primary_metric": "roc_auc",
                       "history_file": str(tmp_path / "artifacts/metrics/history.json")},
    }
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return path


def test_pipeline_moderate_runs(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    result = run_pipeline(cfg, scenario="moderate", make_reports=False, log_monitoring=True)
    summary = result.summary()
    assert "drift" in summary and "optimization" in summary
    # Data + a metrics artifact should now exist.
    assert (tmp_path / "data/reference.csv").exists()
    assert (tmp_path / "artifacts/metrics/moderate_run.json").exists()
    assert (tmp_path / "artifacts/metrics/history.json").exists()


def test_pipeline_severe_triggers_remediation(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    result = run_pipeline(cfg, scenario="severe", make_reports=False, log_monitoring=False)
    assert result.drift.dataset_drift is True
    assert result.optimization.decision.strategy.value != "none"


def test_pipeline_none_scenario_no_remediation(tmp_path):
    cfg = load_config(_write_config(tmp_path))
    result = run_pipeline(cfg, scenario="none", make_reports=False, log_monitoring=False)
    assert result.optimization.decision.strategy.value == "none"
    assert result.optimization.promoted is False
