"""Tests for the shared statistical helpers."""
import numpy as np

from ml_drift.drift.stats import (
    benjamini_hochberg,
    jensen_shannon_divergence,
    normalized_wasserstein,
)


def test_js_zero_for_identical():
    x = np.random.default_rng(0).normal(size=4000)
    assert jensen_shannon_divergence(x, x) < 1e-6


def test_js_bounded_and_increases_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 4000)
    near = rng.normal(0.3, 1, 4000)
    far = rng.normal(5, 1, 4000)
    js_near = jensen_shannon_divergence(ref, near)
    js_far = jensen_shannon_divergence(ref, far)
    assert 0.0 <= js_near <= 1.0
    assert 0.0 <= js_far <= 1.0
    assert js_far > js_near


def test_normalized_wasserstein_scale_free():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(1, 1, 5000)          # 1 sigma shift
    ref_big = ref * 1000
    cur_big = cur * 1000                   # same shift, 1000x scale
    w1 = normalized_wasserstein(ref, cur)
    w2 = normalized_wasserstein(ref_big, cur_big)
    assert abs(w1 - w2) < 0.05             # scale-free
    assert w1 > 0.5


def test_normalized_wasserstein_degenerate():
    assert normalized_wasserstein(np.array([]), np.array([1.0])) == 0.0
    assert normalized_wasserstein(np.array([5.0, 5.0]), np.array([5.0, 5.0])) == 0.0


def test_bh_all_significant():
    reject = benjamini_hochberg([0.001, 0.002, 0.0001], alpha=0.05)
    assert reject.all()


def test_bh_none_significant():
    reject = benjamini_hochberg([0.9, 0.8, 0.95], alpha=0.05)
    assert not reject.any()


def test_bh_is_more_conservative_than_raw():
    # Many features, a few genuinely small p-values mixed with noise.
    rng = np.random.default_rng(2)
    pvals = np.concatenate([rng.uniform(0, 1, 95), [0.001, 0.002, 0.003, 0.004, 0.005]])
    raw_rejections = (pvals < 0.05).sum()
    bh_rejections = benjamini_hochberg(pvals, alpha=0.05).sum()
    assert bh_rejections <= raw_rejections


def test_bh_empty():
    assert benjamini_hochberg([], alpha=0.05).size == 0
