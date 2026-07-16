#!/usr/bin/env python3
"""Generate the reference and current (drifted) datasets to disk.

Usage:
    python scripts/generate_data.py --scenario moderate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_drift.config import load_config
from ml_drift.data.generate import DriftSpec, generate_dataset
from ml_drift.data.loader import save_frame

SCENARIOS = {"none": DriftSpec.none, "moderate": DriftSpec.moderate, "severe": DriftSpec.severe}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="moderate")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    n = cfg["data"]["n_samples"]
    seed = cfg["project"]["random_seed"]

    reference = generate_dataset(n, seed=seed, spec=DriftSpec.none())
    current = generate_dataset(n, seed=seed + 1, spec=SCENARIOS[args.scenario]())

    ref_path = save_frame(reference, cfg.path("reference_data"))
    cur_path = save_frame(current, cfg.path("current_data"))
    print(f"Reference ({len(reference)} rows) -> {ref_path}")
    print(f"Current   ({len(current)} rows, scenario={args.scenario}) -> {cur_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
