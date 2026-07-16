#!/usr/bin/env python3
"""Train the baseline champion model on the reference dataset.

Usage:
    python scripts/train_baseline.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_drift.config import load_config
from ml_drift.data.generate import DriftSpec, generate_dataset
from ml_drift.data.loader import load_frame
from ml_drift.models.train import save_model, train_model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()

    ref_path = cfg.path("reference_data")
    if ref_path.exists():
        reference = load_frame(ref_path)
    else:
        reference = generate_dataset(
            cfg["data"]["n_samples"], seed=cfg["project"]["random_seed"], spec=DriftSpec.none()
        )

    bundle = train_model(
        reference,
        target=cfg["data"]["target"],
        model_type=cfg["model"]["type"],
        params=cfg["model"]["params"],
        test_size=cfg["model"]["test_size"],
        seed=cfg["project"]["random_seed"],
    )
    path = save_model(bundle, cfg.path("model_dir") / "champion.joblib")
    print(f"Trained {bundle.model_type} champion -> {path}")
    print("Validation metrics:")
    print(json.dumps({k: round(v, 4) for k, v in bundle.metrics.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
