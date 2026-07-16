#!/usr/bin/env python3
"""Minimal, self-contained quickstart.

Generates data, trains a champion, injects severe drift, detects it, and runs
post-detection optimization — all in ~20 lines, no config file required.

Run with:  python examples/quickstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_drift.data.generate import DriftSpec, generate_dataset
from ml_drift.drift.detector import detect_dataset_drift
from ml_drift.models.train import train_model
from ml_drift.optimization.optimizer import optimize_after_detection

# 1. Reference data + champion model.
reference = generate_dataset(4000, seed=42, spec=DriftSpec.none())
champion = train_model(reference, model_type="random_forest",
                       params={"n_estimators": 150, "max_depth": 10})
print("Champion validation:", {k: round(v, 3) for k, v in champion.metrics.items()})

# 2. Production data has drifted severely (covariate + concept drift).
current = generate_dataset(4000, seed=99, spec=DriftSpec.severe())

# 3. Detect drift.
drift = detect_dataset_drift(reference, current)
print(f"\nDataset drift: {drift.dataset_drift} "
      f"({drift.n_drifted}/{drift.n_features} features, share={drift.drift_share:.2f})")
print("Drifted features:", drift.drifted_features)

# 4. Optimize post-detection: pick a strategy, retrain, champion/challenger gate.
outcome = optimize_after_detection(champion, reference, current, drift,
                                   primary_metric="roc_auc", strategy="auto")
print("\nRemediation:", outcome.decision.strategy.value, "-", outcome.decision.reason)
print(f"Champion roc_auc on current : {outcome.champion_metric:.3f}")
if outcome.challenger_metric is not None:
    print(f"Challenger roc_auc on current: {outcome.challenger_metric:.3f} "
          f"(Δ {outcome.improvement:+.3f})")
print("Promoted challenger:", outcome.promoted)
