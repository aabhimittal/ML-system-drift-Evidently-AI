"""Native, dependency-light drift detection core.

This module deliberately does *not* depend on Evidently. It computes the same
statistics Evidently uses under the hood, so the system can detect drift, decide
on remediation, and run its tests fully offline. Evidently is layered on top
(``reports.py``) purely for rich, shareable HTML reports.

Per-feature tests
-----------------
* Numeric features: two-sample **Kolmogorov–Smirnov** test (distribution shape)
  and **Population Stability Index** (binned mass shift).
* Categorical features: **Chi-square** test of the category counts and PSI over
  category frequencies.

A feature is flagged as drifted when its statistical test rejects the null
(p-value below threshold) *or* its PSI exceeds the configured threshold. The
dataset is flagged as drifted when the share of drifted features meets
``dataset_drift_share``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .stats import (
    benjamini_hochberg,
    jensen_shannon_divergence,
    jensen_shannon_divergence_categorical,
    normalized_wasserstein,
)


@dataclass
class FeatureDrift:
    """Drift verdict and statistics for a single feature.

    ``js_divergence`` (bounded [0, 1]) and ``wasserstein`` (normalized by the
    reference std) supplement PSI + the hypothesis test; ``wasserstein`` is 0 for
    categorical features. Both default to 0.0 so older call sites remain valid.
    """

    feature: str
    kind: str  # "numerical" | "categorical"
    drifted: bool
    psi: float
    test_name: str
    statistic: float
    p_value: float
    js_divergence: float = 0.0
    wasserstein: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "drifted": self.drifted,
            "psi": round(self.psi, 4),
            "js_divergence": round(self.js_divergence, 4),
            "wasserstein": round(self.wasserstein, 4),
            "test": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 4),
        }


@dataclass
class DriftResult:
    """Aggregate dataset-level drift outcome."""

    dataset_drift: bool
    drift_share: float
    n_features: int
    n_drifted: int
    features: List[FeatureDrift] = field(default_factory=list)

    @property
    def drifted_features(self) -> List[str]:
        return [f.feature for f in self.features if f.drifted]

    def unstable_features(self, psi_threshold: float) -> List[str]:
        """Features whose PSI exceeds ``psi_threshold`` (candidates to stabilise)."""
        return [f.feature for f in self.features if f.psi >= psi_threshold]

    def as_dict(self) -> Dict[str, object]:
        return {
            "dataset_drift": self.dataset_drift,
            "drift_share": round(self.drift_share, 4),
            "n_features": self.n_features,
            "n_drifted": self.n_drifted,
            "drifted_features": self.drifted_features,
            "features": [f.as_dict() for f in self.features],
        }


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Population Stability Index between two 1-D numeric samples.

    PSI = sum( (cur% - ref%) * ln(cur% / ref%) ) over quantile bins of the
    reference. Rule of thumb: <0.1 no shift, 0.1-0.2 minor, >0.2 major shift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0

    # Quantile edges from the reference; widen the outer edges to catch tails.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:  # constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _detect_numeric(
    name: str,
    reference: pd.Series,
    current: pd.Series,
    ks_pvalue_threshold: float,
    psi_threshold: float,
) -> FeatureDrift:
    ref = reference.to_numpy(dtype=float)
    cur = current.to_numpy(dtype=float)
    # Drop non-finite values (NaN / ±inf) that would break the statistics.
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return FeatureDrift(name, "numerical", False, 0.0, "ks", 0.0, 1.0, 0.0, 0.0)
    # A constant reference feature has no distribution to drift *from*; fall back
    # to a mean-shift check so a moved constant is still caught.
    if np.ptp(ref) == 0:
        drifted = bool(not np.array_equal(np.unique(cur), np.unique(ref)))
        wass = normalized_wasserstein(ref, cur)
        return FeatureDrift(name, "numerical", drifted, 0.0, "constant-ref",
                            0.0, 1.0, 0.0, wass)
    ks_stat, p_value = stats.ks_2samp(ref, cur)
    psi = population_stability_index(ref, cur)
    js = jensen_shannon_divergence(ref, cur)
    wass = normalized_wasserstein(ref, cur)
    drifted = bool(p_value < ks_pvalue_threshold or psi >= psi_threshold)
    return FeatureDrift(name, "numerical", drifted, psi, "ks",
                        float(ks_stat), float(p_value), js, wass)


def _detect_categorical(
    name: str,
    reference: pd.Series,
    current: pd.Series,
    chi2_pvalue_threshold: float,
    psi_threshold: float,
) -> FeatureDrift:
    # Cast to string so mixed/typed categories (bools, ints, NaN) compare cleanly,
    # and so an unseen level in the current batch is handled like any other.
    ref_s = reference.dropna().astype(str)
    cur_s = current.dropna().astype(str)
    cats = sorted(set(ref_s.unique()) | set(cur_s.unique()))
    if not cats or (ref_s.empty or cur_s.empty):
        return FeatureDrift(name, "categorical", False, 0.0, "chi2", 0.0, 1.0, 0.0, 0.0)
    ref_counts = np.array([(ref_s == c).sum() for c in cats], dtype=float)
    cur_counts = np.array([(cur_s == c).sum() for c in cats], dtype=float)
    ref_p = ref_counts / max(ref_counts.sum(), 1)
    cur_p = cur_counts / max(cur_counts.sum(), 1)
    psi = float(np.sum((cur_p + 1e-6 - (ref_p + 1e-6)) * np.log((cur_p + 1e-6) / (ref_p + 1e-6))))
    js = jensen_shannon_divergence_categorical(ref_p, cur_p)

    # Chi-square test on the contingency of counts. Guard degenerate tables.
    p_value, stat = 1.0, 0.0
    table = np.vstack([ref_counts, cur_counts])
    if table.shape[1] > 1 and table.sum() > 0 and (table.sum(axis=0) > 0).all():
        try:
            stat, p_value, _, _ = stats.chi2_contingency(table + 1e-6)
        except ValueError:  # pragma: no cover - degenerate
            p_value, stat = 1.0, 0.0
    drifted = bool(p_value < chi2_pvalue_threshold or psi >= psi_threshold)
    return FeatureDrift(name, "categorical", drifted, psi, "chi2",
                        float(stat), float(p_value), js, 0.0)


def detect_dataset_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Optional[List[str]] = None,
    ks_pvalue_threshold: float = 0.05,
    chi2_pvalue_threshold: float = 0.05,
    psi_threshold: float = 0.2,
    dataset_drift_share: float = 0.5,
    correction: str = "none",
    fdr_alpha: float = 0.05,
) -> DriftResult:
    """Run per-feature drift tests and aggregate to a dataset-level verdict.

    Robust to industrial edge cases: it compares only the columns present in
    *both* frames, coerces the current column to the reference's numeric dtype
    when needed (so a column that arrives as strings is still tested), and
    tolerates empty/constant/all-NaN columns and unseen categorical levels.

    Parameters
    ----------
    correction: ``"none"`` (default) uses each test's raw p-value threshold;
        ``"bh"`` applies a Benjamini–Hochberg false-discovery-rate correction
        across all features (recommended when the feature count is large).
    fdr_alpha: target false-discovery rate when ``correction == "bh"``.
    """
    if features is None:
        # Compare on columns present in both frames, excluding common label cols.
        skip = {"target", "prediction", "prediction_proba"}
        features = [c for c in reference.columns if c in current.columns and c not in skip]
    else:
        features = [c for c in features if c in reference.columns and c in current.columns]

    results: List[FeatureDrift] = []
    for name in features:
        ref_col, cur_col = reference[name], current[name]
        if pd.api.types.is_numeric_dtype(ref_col):
            # If the current column is not numeric (schema drift / mixed types),
            # coerce it — unparseable values become NaN and are dropped downstream.
            if not pd.api.types.is_numeric_dtype(cur_col):
                cur_col = pd.to_numeric(cur_col, errors="coerce")
            results.append(
                _detect_numeric(name, ref_col, cur_col, ks_pvalue_threshold, psi_threshold)
            )
        else:
            results.append(
                _detect_categorical(
                    name, ref_col, cur_col, chi2_pvalue_threshold, psi_threshold
                )
            )

    # Optional multiple-testing correction across all per-feature p-values.
    if correction == "bh" and results:
        reject = benjamini_hochberg([r.p_value for r in results], alpha=fdr_alpha)
        for r, rej in zip(results, reject):
            r.drifted = bool(rej or r.psi >= psi_threshold)

    n_features = len(results)
    n_drifted = sum(1 for r in results if r.drifted)
    drift_share = n_drifted / n_features if n_features else 0.0
    dataset_drift = drift_share >= dataset_drift_share
    return DriftResult(dataset_drift, drift_share, n_features, n_drifted, results)
