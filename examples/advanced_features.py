#!/usr/bin/env python3
"""Demonstrate the advanced, production-oriented features.

Run with:  python examples/advanced_features.py

Shows, on a severe-drift scenario:
  1. Schema / data-contract validation   (catch hard breaks before scoring)
  2. Impact-weighted drift                (which drift actually matters to the model)
  3. Unsupervised prediction drift        (early warning without labels)
  4. Segmented drift                      (find the drifting pocket)
  5. Cost/value-aware remediation gate    (retrain only when it pays off)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ml_drift.data.generate import DriftSpec, generate_dataset
from ml_drift.drift.advanced import (
    detect_prediction_drift,
    impact_weighted_drift,
    model_feature_importances,
    segmented_drift,
)
from ml_drift.drift.detector import detect_dataset_drift
from ml_drift.drift.schema import validate_batch
from ml_drift.models.predict import predict_proba
from ml_drift.models.train import train_model
from ml_drift.optimization.policy import should_promote

reference = generate_dataset(4000, seed=42, spec=DriftSpec.none())
current = generate_dataset(4000, seed=99, spec=DriftSpec.severe())
champion = train_model(reference, params={"n_estimators": 150, "max_depth": 10})

print("=" * 70)
print("1. SCHEMA / DATA-CONTRACT VALIDATION")
print("=" * 70)
broken = current.copy()
broken.loc[broken.index[:200], "region"] = "atlantis"     # unseen category
broken["amount"] = broken["amount"].astype(str)           # dtype change
report = validate_batch(reference, broken, ignore=["target"])
print(f"contract ok={report.ok}  errors={report.n_errors}  warnings={report.n_warnings}")
for v in report.violations[:4]:
    print(f"  [{v.severity:7}] {v.kind:16} {v.column}: {v.detail}")

print("\n" + "=" * 70)
print("2. IMPACT-WEIGHTED DRIFT  (PSI x model importance)")
print("=" * 70)
drift = detect_dataset_drift(reference, current)
importances = model_feature_importances(champion)
iwd = impact_weighted_drift(drift, importances)
print(f"total impact score = {iwd.total_impact:.3f}   top feature = {iwd.top_feature}")
print(f"{'feature':10} {'psi':>7} {'importance':>11} {'impact':>8}")
for r in iwd.rankings[:5]:
    print(f"{r.feature:10} {r.psi:7.3f} {r.importance:11.3f} {r.impact:8.3f}")

print("\n" + "=" * 70)
print("3. UNSUPERVISED PREDICTION DRIFT  (no labels needed)")
print("=" * 70)
ref_scores = predict_proba(champion, reference)
cur_scores = predict_proba(champion, current)
pdrift = detect_prediction_drift(ref_scores, cur_scores)
print(f"drifted={pdrift.drifted}  psi={pdrift.psi:.3f}  "
      f"js={pdrift.js_divergence:.3f}  ks_stat={pdrift.ks_statistic:.3f}  "
      f"mean_shift={pdrift.mean_shift:+.3f}")

print("\n" + "=" * 70)
print("4. SEGMENTED DRIFT  (find the drifting pocket)")
print("=" * 70)
seg = segmented_drift(reference, current, "region", min_segment_size=100)
for name, res in seg.segments.items():
    print(f"  region={name:6}  drift_share={res.drift_share:.2f}  "
          f"dataset_drift={res.dataset_drift}")
if seg.worst_segment:
    print(f"  worst segment: {seg.worst_segment[0]} (share={seg.worst_segment[1]:.2f})")

print("\n" + "=" * 70)
print("5. COST/VALUE-AWARE REMEDIATION GATE")
print("=" * 70)
# Suppose a challenger improves ROC-AUC by 0.08; is retraining worth it?
for cost in (50, 5000):
    v = should_promote(0.62, 0.70, value_per_metric_point=10000, retrain_cost=cost)
    print(f"  retrain_cost=${cost:<5} -> promote={v.promote}  ({v.reason})")
