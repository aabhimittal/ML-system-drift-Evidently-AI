#!/usr/bin/env python3
"""Detect drift between reference and current data + emit Evidently reports.

Usage:
    python scripts/detect_drift.py
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
from ml_drift.drift.reports import EVIDENTLY_AVAILABLE, generate_all_reports
from ml_drift.models.predict import score_frame
from ml_drift.models.train import load_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-reports", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    reference = load_frame(cfg.path("reference_data"))
    current = load_frame(cfg.path("current_data"))

    dcfg = cfg["drift"]
    drift = detect_dataset_drift(
        reference, current,
        ks_pvalue_threshold=dcfg["ks_pvalue_threshold"],
        chi2_pvalue_threshold=dcfg["chi2_pvalue_threshold"],
        psi_threshold=dcfg["psi_threshold"],
        dataset_drift_share=dcfg["dataset_drift_share"],
    )
    print(json.dumps(drift.as_dict(), indent=2))

    if not args.no_reports:
        model_path = cfg.path("model_dir") / "champion.joblib"
        if model_path.exists():
            champion = load_model(model_path)
            reference = score_frame(champion, reference)
            current = score_frame(champion, current)
            numeric, categorical = champion.numeric_features, champion.categorical_features
        else:
            numeric = categorical = None
        reports = generate_all_reports(
            reference, current, cfg.path("reports_dir"),
            target=cfg["data"]["target"], numeric=numeric, categorical=categorical,
        )
        if EVIDENTLY_AVAILABLE:
            print("\nEvidently reports:")
            for name, path in reports.items():
                print(f"  - {name}: {path}")
        else:
            print("\nEvidently not installed; skipped HTML reports "
                  "(native drift decision above is authoritative).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
