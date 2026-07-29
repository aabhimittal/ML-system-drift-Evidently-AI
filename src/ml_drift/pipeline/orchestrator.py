"""End-to-end pipeline: data -> train -> detect -> report -> optimize -> log.

This ties every layer together into a single, auditable run. It is intentionally
written so each step can also be run standalone via the ``scripts/`` entry points.

Flow
----
1. Load (or generate) the reference and current batches.
2. Train (or load) the champion model on the reference.
3. Score the current batch with the champion (for Evidently reports).
4. Detect data drift with the native detector.
5. Generate Evidently HTML reports (skipped gracefully if Evidently absent).
6. Run post-detection optimization (remediation + champion/challenger).
7. Persist the (possibly promoted) model, metrics and monitoring history.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..config import Config, load_config
from ..data.generate import DriftSpec, generate_dataset
from ..data.loader import load_frame, save_frame
from ..drift.detector import DriftResult, detect_dataset_drift
from ..drift.advanced import (
    detect_prediction_drift,
    impact_weighted_drift,
    model_feature_importances,
    segmented_drift,
)
from ..drift.schema import validate_batch
from ..drift.reports import EVIDENTLY_AVAILABLE, generate_all_reports
from ..models.predict import predict_proba, score_frame
from ..models.train import ModelBundle, save_model, train_model
from ..monitoring.dashboard import append_history
from ..optimization.optimizer import OptimizationOutcome, optimize_after_detection

# Named drift scenarios usable from the CLI / tests.
SCENARIOS = {
    "none": DriftSpec.none,
    "moderate": DriftSpec.moderate,
    "severe": DriftSpec.severe,
}


@dataclass
class PipelineResult:
    drift: DriftResult
    optimization: OptimizationOutcome
    champion_before: Dict[str, float]
    reports: Dict[str, Optional[str]] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    analytics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "drift": self.drift.as_dict(),
            "optimization": self.optimization.summary(),
            "champion_before": {k: round(v, 4) for k, v in self.champion_before.items()},
            "reports": self.reports,
            "analytics": self.analytics,
            "evidently_available": EVIDENTLY_AVAILABLE,
        }


def _get_or_train_champion(
    cfg: Config, reference: pd.DataFrame, retrain: bool
) -> ModelBundle:
    model_path = cfg.path("model_dir") / "champion.joblib"
    if model_path.exists() and not retrain:
        from ..models.train import load_model

        return load_model(model_path)
    bundle = train_model(
        reference,
        target=cfg["data"]["target"],
        model_type=cfg["model"]["type"],
        params=cfg["model"]["params"],
        test_size=cfg["model"]["test_size"],
        seed=cfg["project"]["random_seed"],
    )
    save_model(bundle, model_path)
    return bundle


def run_pipeline(
    cfg: Optional[Config] = None,
    scenario: str = "moderate",
    *,
    regenerate: bool = True,
    retrain_champion: bool = True,
    make_reports: bool = True,
    log_monitoring: bool = True,
) -> PipelineResult:
    """Execute the full pipeline and return a :class:`PipelineResult`."""
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    seed = cfg["project"]["random_seed"]
    target = cfg["data"]["target"]
    n = cfg["data"]["n_samples"]

    # 1. Data ---------------------------------------------------------------
    ref_path, cur_path = cfg.path("reference_data"), cfg.path("current_data")
    if regenerate or not ref_path.exists() or not cur_path.exists():
        spec_factory = SCENARIOS.get(scenario, DriftSpec.moderate)
        reference = generate_dataset(n, seed=seed, spec=DriftSpec.none())
        current = generate_dataset(n, seed=seed + 1, spec=spec_factory())
        save_frame(reference, ref_path)
        save_frame(current, cur_path)
    else:
        reference = load_frame(ref_path)
        current = load_frame(cur_path)

    # 2. Champion -----------------------------------------------------------
    champion = _get_or_train_champion(cfg, reference, retrain_champion)
    champion_before = dict(champion.metrics)

    # 3. Score current for reports (predictions alongside ground truth) -----
    reference_scored = score_frame(champion, reference)
    current_scored = score_frame(champion, current)

    # 4. Detect drift (native core) ----------------------------------------
    dcfg = cfg["drift"]
    drift = detect_dataset_drift(
        reference,
        current,
        ks_pvalue_threshold=dcfg["ks_pvalue_threshold"],
        chi2_pvalue_threshold=dcfg["chi2_pvalue_threshold"],
        psi_threshold=dcfg["psi_threshold"],
        dataset_drift_share=dcfg["dataset_drift_share"],
        correction=dcfg.get("correction", "none"),
        fdr_alpha=dcfg.get("fdr_alpha", 0.05),
    )

    # 4b. Advanced analytics (schema, impact-weighting, prediction & segment) -
    acfg = cfg.get("advanced", {}) or {}
    analytics: Dict[str, Any] = {}
    if acfg.get("enable_schema_validation", True):
        analytics["schema"] = validate_batch(
            reference, current, ignore=[target, "prediction", "prediction_proba"]
        ).as_dict()
    if acfg.get("enable_impact_weighting", True):
        try:
            importances = model_feature_importances(champion)
            analytics["impact_weighted_drift"] = impact_weighted_drift(
                drift, importances
            ).as_dict()
        except Exception:  # pragma: no cover - never let analytics break the run
            pass
    if acfg.get("enable_prediction_drift", True):
        ref_scores = predict_proba(champion, reference)
        cur_scores = predict_proba(champion, current)
        analytics["prediction_drift"] = detect_prediction_drift(
            ref_scores, cur_scores,
            psi_threshold=acfg.get("prediction_psi_threshold", dcfg["psi_threshold"]),
        ).as_dict()
    seg_col = acfg.get("segment_column")
    if seg_col and seg_col in reference.columns and seg_col in current.columns:
        analytics["segmented_drift"] = segmented_drift(
            reference, current, seg_col,
            min_segment_size=acfg.get("min_segment_size", 100),
            psi_threshold=dcfg["psi_threshold"],
            dataset_drift_share=dcfg["dataset_drift_share"],
        ).as_dict()

    # 5. Evidently reports (optional) --------------------------------------
    reports: Dict[str, Optional[str]] = {}
    if make_reports:
        reports = generate_all_reports(
            reference_scored,
            current_scored,
            cfg.path("reports_dir"),
            target=target,
            prediction="prediction",
            numeric=champion.numeric_features,
            categorical=champion.categorical_features,
            prefix=scenario,
        )

    # 6. Post-detection optimization ---------------------------------------
    ocfg = cfg["optimization"]
    outcome = optimize_after_detection(
        champion,
        reference,
        current,
        drift,
        target=target,
        primary_metric=cfg["monitoring"]["primary_metric"],
        strategy=ocfg["strategy"],
        model_type=cfg["model"]["type"],
        model_params=cfg["model"]["params"],
        min_drift_share_to_retrain=ocfg["min_drift_share_to_retrain"],
        performance_drop_to_retrain=ocfg["performance_drop_to_retrain"],
        psi_drop_feature_threshold=ocfg["psi_drop_feature_threshold"],
        rolling_window_size=ocfg["rolling_window_size"],
        recent_sample_weight=ocfg["recent_sample_weight"],
        champion_challenger=ocfg["champion_challenger"],
        min_improvement=ocfg["min_improvement"],
        value_per_metric_point=ocfg.get("value_per_metric_point"),
        retrain_cost=ocfg.get("retrain_cost", 0.0),
        seed=seed,
    )

    # 7. Persist model + metrics + history ---------------------------------
    paths: Dict[str, str] = {}
    if outcome.promoted:
        best_path = cfg.path("model_dir") / "champion.joblib"
        save_model(outcome.best_bundle, best_path)
        paths["promoted_model"] = str(best_path)

    metrics_path = cfg.path("metrics_dir") / f"{scenario}_run.json"
    result = PipelineResult(
        drift=drift,
        optimization=outcome,
        champion_before=champion_before,
        reports=reports,
        paths=paths,
        analytics=analytics,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(result.summary(), fh, indent=2, default=str)
    paths["metrics"] = str(metrics_path)

    if log_monitoring:
        record = {
            "scenario": scenario,
            "drift_share": drift.drift_share,
            "dataset_drift": drift.dataset_drift,
            "n_drifted": drift.n_drifted,
            "strategy": outcome.decision.strategy.value,
            "champion_metric": outcome.champion_metric,
            "challenger_metric": outcome.challenger_metric,
            "promoted": outcome.promoted,
            "primary_metric": outcome.primary_metric,
        }
        history_file = Path(cfg["monitoring"]["history_file"])
        if not history_file.is_absolute():
            history_file = cfg.root / history_file
        append_history(history_file, record)

    return result


def _print_summary(result: PipelineResult) -> None:
    s = result.summary()
    d, o = s["drift"], s["optimization"]
    print("\n=== DRIFT DETECTION ===")
    print(f"  dataset_drift : {d['dataset_drift']}  (share={d['drift_share']}, "
          f"{d['n_drifted']}/{d['n_features']} features)")
    print(f"  drifted       : {', '.join(d['drifted_features']) or 'none'}")
    print("\n=== POST-DETECTION OPTIMIZATION ===")
    print(f"  strategy      : {o['strategy']}  ({o['reason']})")
    print(f"  {o['primary_metric']:<13}: champion={o['champion_metric']} "
          f"challenger={o['challenger_metric']}  (Δ={o['improvement']})")
    print(f"  decision      : {'PROMOTED challenger' if o['promoted'] else 'kept champion'}")

    a = s.get("analytics", {})
    if a:
        print("\n=== ADVANCED ANALYTICS ===")
        if "schema" in a:
            sc = a["schema"]
            print(f"  schema        : ok={sc['ok']} "
                  f"(errors={sc['n_errors']}, warnings={sc['n_warnings']})")
        if "impact_weighted_drift" in a:
            iwd = a["impact_weighted_drift"]
            print(f"  impact drift  : total={iwd['total_impact']} "
                  f"top_feature={iwd['top_feature']}")
        if "prediction_drift" in a:
            pdd = a["prediction_drift"]
            print(f"  pred. drift   : drifted={pdd['drifted']} "
                  f"(psi={pdd['psi']}, js={pdd['js_divergence']}, "
                  f"mean_shift={pdd['mean_shift']})  [unsupervised]")
        if "segmented_drift" in a and a["segmented_drift"].get("worst_segment"):
            ws = a["segmented_drift"]["worst_segment"]
            print(f"  worst segment : {a['segmented_drift']['segment_column']}={ws[0]} "
                  f"(share={ws[1]})")

    print(f"\n  Evidently reports: {'generated' if s['evidently_available'] else 'skipped (evidently not installed)'}")
    for name, path in s["reports"].items():
        if path:
            print(f"    - {name}: {path}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the end-to-end drift + optimization pipeline.")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="moderate",
                        help="drift scenario to simulate for the current batch")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--no-reports", action="store_true", help="skip Evidently reports")
    parser.add_argument("--no-regenerate", action="store_true",
                        help="reuse existing data/*.csv instead of regenerating")
    parser.add_argument("--log-monitoring", action="store_true",
                        help="append this run to the monitoring history")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    result = run_pipeline(
        cfg,
        scenario=args.scenario,
        regenerate=not args.no_regenerate,
        make_reports=not args.no_reports,
        log_monitoring=args.log_monitoring,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
