#!/usr/bin/env python3
"""Run post-detection optimization (remediation) against the current batch.

Detects drift, then applies the configured remediation strategy with a
champion/challenger gate. Prints the decision and whether a new model was
promoted.

Usage:
    python scripts/optimize.py --strategy auto
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_drift.config import load_config
from ml_drift.data.loader import load_frame
from ml_drift.drift.detector import detect_dataset_drift
from ml_drift.models.train import load_model, save_model
from ml_drift.optimization.optimizer import optimize_after_detection


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--strategy", default=None,
                    help="override config strategy (auto|retrain|reweight|feature_stabilize|none)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    reference = load_frame(cfg.path("reference_data"))
    current = load_frame(cfg.path("current_data"))
    champion = load_model(cfg.path("model_dir") / "champion.joblib")

    dcfg, ocfg = cfg["drift"], cfg["optimization"]
    drift = detect_dataset_drift(
        reference, current,
        ks_pvalue_threshold=dcfg["ks_pvalue_threshold"],
        chi2_pvalue_threshold=dcfg["chi2_pvalue_threshold"],
        psi_threshold=dcfg["psi_threshold"],
        dataset_drift_share=dcfg["dataset_drift_share"],
    )

    outcome = optimize_after_detection(
        champion, reference, current, drift,
        target=cfg["data"]["target"],
        primary_metric=cfg["monitoring"]["primary_metric"],
        strategy=args.strategy or ocfg["strategy"],
        model_type=cfg["model"]["type"],
        model_params=cfg["model"]["params"],
        min_drift_share_to_retrain=ocfg["min_drift_share_to_retrain"],
        performance_drop_to_retrain=ocfg["performance_drop_to_retrain"],
        psi_drop_feature_threshold=ocfg["psi_drop_feature_threshold"],
        rolling_window_size=ocfg["rolling_window_size"],
        recent_sample_weight=ocfg["recent_sample_weight"],
        champion_challenger=ocfg["champion_challenger"],
        min_improvement=ocfg["min_improvement"],
        seed=cfg["project"]["random_seed"],
    )
    print(json.dumps(outcome.summary(), indent=2))

    if outcome.promoted:
        path = save_model(outcome.best_bundle, cfg.path("model_dir") / "champion.joblib")
        print(f"\nPromoted challenger -> {path}")
    else:
        print("\nChampion retained (challenger did not clear the gate).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
