#!/usr/bin/env python3
"""Run the full end-to-end pipeline (thin wrapper around the orchestrator).

Usage:
    python scripts/run_pipeline.py --scenario severe --log-monitoring
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_drift.pipeline.orchestrator import main

if __name__ == "__main__":
    raise SystemExit(main())
