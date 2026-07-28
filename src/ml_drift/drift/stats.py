"""Pure statistical helpers shared by the detector and advanced analytics.

Kept dependency-light (numpy + scipy) and free of any project imports so both
``detector.py`` and ``advanced.py`` can use them without circular imports.

Included
--------
* :func:`jensen_shannon_divergence` — bounded [0, 1] distribution distance.
* :func:`normalized_wasserstein`     — earth-mover distance scaled by reference std.
* :func:`benjamini_hochberg`         — false-discovery-rate multiple-testing control.

Why these matter industrially
-----------------------------
* JS divergence is symmetric and **bounded**, so it is safe to threshold and to
  average across features (unlike PSI, which is unbounded and blows up on rare bins).
* Normalized Wasserstein is **scale-free**, so a shift is comparable across a
  feature measured in dollars and one measured in milliseconds.
* With hundreds of features, raw per-feature p-values produce a flood of false
  drift alarms; Benjamini–Hochberg controls the expected false-discovery rate.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats

_EPS = 1e-12


def _clean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return a


def _reference_bin_probs(
    reference: np.ndarray, current: np.ndarray, bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ref_probs, cur_probs) over quantile bins of the reference."""
    reference, current = _clean(reference), _clean(current)
    if reference.size == 0 or current.size == 0:
        return np.array([1.0]), np.array([1.0])
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return np.array([1.0]), np.array([1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_p = ref_counts / max(ref_counts.sum(), 1)
    cur_p = cur_counts / max(cur_counts.sum(), 1)
    return ref_p, cur_p


def _js_from_probs(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence (base-2, in [0, 1]) between two prob vectors."""
    p = np.asarray(p, dtype=float) + _EPS
    q = np.asarray(q, dtype=float) + _EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    js = 0.5 * kl_pm + 0.5 * kl_qm
    # Numerical guard: JS is in [0, 1] for base-2 logs.
    return float(min(max(js, 0.0), 1.0))


def jensen_shannon_divergence(
    reference: np.ndarray, current: np.ndarray, bins: int = 20
) -> float:
    """JS divergence between two numeric samples via reference quantile bins."""
    ref_p, cur_p = _reference_bin_probs(reference, current, bins)
    return _js_from_probs(ref_p, cur_p)


def jensen_shannon_divergence_categorical(ref_probs, cur_probs) -> float:
    """JS divergence between two aligned categorical probability vectors."""
    return _js_from_probs(np.asarray(ref_probs, float), np.asarray(cur_probs, float))


def normalized_wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    """Wasserstein-1 distance normalized by the reference standard deviation.

    Dividing by the reference spread makes the metric scale-free and comparable
    across heterogeneous features. Returns 0.0 for degenerate inputs.
    """
    reference, current = _clean(reference), _clean(current)
    if reference.size == 0 or current.size == 0:
        return 0.0
    scale = np.std(reference)
    if not np.isfinite(scale) or scale < _EPS:
        scale = 1.0
    return float(stats.wasserstein_distance(reference, current) / scale)


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR procedure.

    Returns a boolean array (same length/order as ``pvalues``) where True marks a
    rejected null (i.e. a statistically significant drift) while controlling the
    expected false-discovery rate at ``alpha``.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresholds
    reject = np.zeros(n, dtype=bool)
    if below.any():
        # Largest rank k where p_(k) <= k/n * alpha; reject all up to k.
        k_max = np.max(np.where(below)[0])
        reject_sorted = np.zeros(n, dtype=bool)
        reject_sorted[: k_max + 1] = True
        reject[order] = reject_sorted
    return reject
