"""Advanced, production-oriented drift analytics.

These go beyond "did the input distribution move?" to answer the questions an
on-call ML engineer actually has:

* **Which drift actually matters?** — :func:`impact_weighted_drift` weights each
  feature's distribution shift by how much the model *relies* on that feature, so
  a big shift in an ignored feature ranks below a small shift in a pivotal one.
* **Is something wrong before labels arrive?** — :func:`detect_prediction_drift`
  watches the model's own output/score distribution, an unsupervised early-warning
  signal for the common industrial case where ground truth is delayed or missing.
* **Is drift hiding inside a segment?** — :func:`segmented_drift` runs detection
  per slice (e.g. per region) so a severe pocket isn't masked by a calm average.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .detector import DriftResult, detect_dataset_drift, population_stability_index
from .stats import jensen_shannon_divergence


# --------------------------------------------------------------------------- #
# 1. Impact-weighted drift: distribution shift × model reliance
# --------------------------------------------------------------------------- #
def model_feature_importances(bundle) -> Dict[str, float]:
    """Extract per-original-feature importances from a trained ``ModelBundle``.

    The pipeline one-hot-expands categoricals, so importances are summed back to
    the original column. Works with tree models (``feature_importances_``) and
    linear models (``|coef_|``). Returns importances normalised to sum to 1.
    """
    pipe = bundle.pipeline
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:  # pragma: no cover - unusual estimator
        return {f: 1.0 / len(bundle.feature_names) for f in bundle.feature_names}

    try:
        transformed_names = list(pre.get_feature_names_out())
    except Exception:  # pragma: no cover - very old sklearn
        return {f: 1.0 / len(bundle.feature_names) for f in bundle.feature_names}

    # Map each transformed column back to its original feature and sum.
    agg: Dict[str, float] = {f: 0.0 for f in bundle.feature_names}
    originals = bundle.feature_names
    for tname, imp in zip(transformed_names, raw):
        # ColumnTransformer names look like "num__amount" or "cat__region_north".
        stripped = tname.split("__", 1)[-1]
        matched = None
        for orig in originals:
            if stripped == orig or stripped.startswith(orig + "_"):
                matched = orig
                break
        if matched is None:
            matched = stripped
        agg[matched] = agg.get(matched, 0.0) + float(imp)

    total = sum(agg.values())
    if total <= 0:
        return {f: 1.0 / len(originals) for f in originals}
    return {k: v / total for k, v in agg.items()}


@dataclass
class ImpactRanking:
    feature: str
    psi: float
    importance: float
    impact: float  # psi * importance

    def as_dict(self) -> Dict[str, object]:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "importance": round(self.importance, 4),
            "impact": round(self.impact, 4),
        }


@dataclass
class ImpactWeightedDrift:
    total_impact: float
    rankings: List[ImpactRanking] = field(default_factory=list)

    @property
    def top_feature(self) -> Optional[str]:
        return self.rankings[0].feature if self.rankings else None

    def as_dict(self) -> Dict[str, object]:
        return {
            "total_impact": round(self.total_impact, 4),
            "top_feature": self.top_feature,
            "rankings": [r.as_dict() for r in self.rankings],
        }


def impact_weighted_drift(
    drift: DriftResult, feature_importances: Dict[str, float]
) -> ImpactWeightedDrift:
    """Rank drifted features by ``PSI × model importance``.

    ``total_impact`` is the importance-weighted sum of PSI across features — a
    single number that rises only when the model's *influential* inputs drift,
    letting you ignore benign drift in features the model barely uses.
    """
    rankings: List[ImpactRanking] = []
    total = 0.0
    for fd in drift.features:
        imp = float(feature_importances.get(fd.feature, 0.0))
        impact = fd.psi * imp
        total += impact
        rankings.append(ImpactRanking(fd.feature, fd.psi, imp, impact))
    rankings.sort(key=lambda r: r.impact, reverse=True)
    return ImpactWeightedDrift(total_impact=total, rankings=rankings)


# --------------------------------------------------------------------------- #
# 2. Unsupervised prediction drift (delayed / missing labels)
# --------------------------------------------------------------------------- #
@dataclass
class PredictionDrift:
    drifted: bool
    psi: float
    js_divergence: float
    ks_statistic: float
    ks_pvalue: float
    mean_shift: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "drifted": self.drifted,
            "psi": round(self.psi, 4),
            "js_divergence": round(self.js_divergence, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_pvalue": round(self.ks_pvalue, 4),
            "mean_shift": round(self.mean_shift, 4),
        }


def detect_prediction_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    psi_threshold: float = 0.2,
    ks_stat_threshold: float = 0.1,
) -> PredictionDrift:
    """Detect drift in the model's output score distribution — no labels needed.

    When ground truth is delayed, a shift in the predicted-probability
    distribution is the earliest available signal that the world has changed.

    The decision is deliberately based on **effect size** (PSI and the KS
    *statistic* magnitude), not the KS p-value: at production batch sizes the KS
    p-value rejects on statistically-tiny, operationally-irrelevant differences
    (the classic "everything is significant at n=100k" trap). The p-value is
    still reported for reference.
    """
    ref = np.asarray(reference_scores, dtype=float)
    cur = np.asarray(current_scores, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return PredictionDrift(False, 0.0, 0.0, 0.0, 1.0, 0.0)
    psi = population_stability_index(ref, cur)
    js = jensen_shannon_divergence(ref, cur)
    ks_stat, ks_p = stats.ks_2samp(ref, cur)
    mean_shift = float(np.mean(cur) - np.mean(ref))
    drifted = bool(psi >= psi_threshold or ks_stat >= ks_stat_threshold)
    return PredictionDrift(drifted, float(psi), float(js), float(ks_stat),
                           float(ks_p), mean_shift)


# --------------------------------------------------------------------------- #
# 3. Segmented drift (find the drifting pocket)
# --------------------------------------------------------------------------- #
@dataclass
class SegmentedDrift:
    segment_column: str
    segments: Dict[str, DriftResult] = field(default_factory=dict)

    @property
    def worst_segment(self) -> Optional[Tuple[str, float]]:
        if not self.segments:
            return None
        worst = max(self.segments.items(), key=lambda kv: kv[1].drift_share)
        return worst[0], worst[1].drift_share

    def as_dict(self) -> Dict[str, object]:
        return {
            "segment_column": self.segment_column,
            "worst_segment": self.worst_segment,
            "segments": {k: v.as_dict() for k, v in self.segments.items()},
        }


def segmented_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    segment_column: str,
    features: Optional[List[str]] = None,
    min_segment_size: int = 100,
    **detect_kwargs,
) -> SegmentedDrift:
    """Run drift detection independently within each level of ``segment_column``.

    Segments smaller than ``min_segment_size`` in either frame are skipped (too
    little data for a reliable verdict). Aggregate drift can look calm while a
    single segment is severely drifted — this surfaces that pocket.
    """
    if segment_column not in reference.columns or segment_column not in current.columns:
        raise KeyError(f"segment_column '{segment_column}' not found in both frames")

    levels = sorted(set(reference[segment_column].dropna().unique())
                    & set(current[segment_column].dropna().unique()))
    if features is None:
        skip = {"target", "prediction", "prediction_proba", segment_column}
        features = [c for c in reference.columns if c in current.columns and c not in skip]

    out: Dict[str, DriftResult] = {}
    for level in levels:
        ref_seg = reference[reference[segment_column] == level]
        cur_seg = current[current[segment_column] == level]
        if len(ref_seg) < min_segment_size or len(cur_seg) < min_segment_size:
            continue
        out[str(level)] = detect_dataset_drift(
            ref_seg, cur_seg, features=features, **detect_kwargs
        )
    return SegmentedDrift(segment_column=segment_column, segments=out)
