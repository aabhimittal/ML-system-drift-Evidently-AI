"""Synthetic dataset generation with controllable drift injection.

The system is designed to be fully reproducible and offline: instead of
downloading a dataset we *generate* one whose drift we can control precisely.
This makes drift detection and the post-detection optimization measurable.

Data model
----------
* Six numeric features drawn from Gaussian / log-normal distributions.
* Three categorical features drawn from fixed category probabilities.
* A binary ``target`` produced by a logistic model over the features, so a
  classifier can genuinely learn the reference relationship.

Two kinds of drift can be injected via :class:`DriftSpec`:

* **Covariate (data) drift** — the *inputs* shift: numeric means/scales move and
  categorical probabilities re-weight. The learned relationship still holds, but
  the model sees inputs it was not trained on.
* **Concept drift** — the *relationship* between inputs and target changes
  (the logistic coefficients rotate), so yesterday's model is simply wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

NUMERIC_FEATURES: List[str] = ["amount", "tenure", "latency", "score", "balance", "age"]
CATEGORICAL_FEATURES: List[str] = ["region", "channel", "device"]

# Reference category probabilities for each categorical feature.
_CATEGORIES: Dict[str, Dict[str, float]] = {
    "region": {"north": 0.30, "south": 0.30, "east": 0.20, "west": 0.20},
    "channel": {"web": 0.55, "mobile": 0.35, "store": 0.10},
    "device": {"ios": 0.40, "android": 0.45, "desktop": 0.15},
}

# "True" logistic coefficients for the reference concept. Numeric contributions
# are applied to standardized features; categoricals add a per-level offset.
_NUM_COEFS = np.array([0.9, -0.7, 0.5, 1.1, -0.4, 0.3])
_CAT_COEFS: Dict[str, Dict[str, float]] = {
    "region": {"north": 0.20, "south": -0.10, "east": 0.35, "west": -0.30},
    "channel": {"web": 0.10, "mobile": -0.25, "store": 0.40},
    "device": {"ios": 0.15, "android": -0.15, "desktop": 0.05},
}
_INTERCEPT = -0.15


@dataclass
class DriftSpec:
    """Description of the drift to inject when generating a batch.

    Attributes
    ----------
    numeric_mean_shift:
        Per-feature additive shift expressed in units of the feature's own
        standard deviation (e.g. ``{"amount": 1.5}`` moves ``amount`` up by
        1.5 sigma). Missing features default to no shift.
    numeric_scale:
        Per-feature multiplicative scale on the standard deviation
        (``> 1`` widens, ``< 1`` narrows the distribution).
    categorical_shift:
        Per-feature mapping of category -> additive weight applied to the
        reference probabilities before renormalisation (covariate drift).
    concept_rotation:
        Magnitude in ``[0, 1]`` of concept drift. ``0`` keeps the reference
        relationship; ``1`` fully rotates the logistic coefficients, decoupling
        the model from the new labels.
    """

    numeric_mean_shift: Dict[str, float] = field(default_factory=dict)
    numeric_scale: Dict[str, float] = field(default_factory=dict)
    categorical_shift: Dict[str, Dict[str, float]] = field(default_factory=dict)
    concept_rotation: float = 0.0

    @classmethod
    def none(cls) -> "DriftSpec":
        """A no-op spec: identical distribution to the reference."""
        return cls()

    @classmethod
    def moderate(cls) -> "DriftSpec":
        """A realistic 'production has drifted' scenario (covariate + mild concept)."""
        return cls(
            numeric_mean_shift={"amount": 1.4, "latency": 1.1, "score": -0.9},
            numeric_scale={"balance": 1.6, "latency": 1.3},
            categorical_shift={
                "channel": {"mobile": 0.35, "web": -0.20},
                "device": {"android": 0.25, "desktop": -0.10},
            },
            concept_rotation=0.25,
        )

    @classmethod
    def severe(cls) -> "DriftSpec":
        """A strong drift scenario that should force remediation/retraining."""
        return cls(
            numeric_mean_shift={"amount": 2.5, "latency": 2.0, "score": -1.8, "age": 1.2},
            numeric_scale={"balance": 2.2, "latency": 1.8, "amount": 1.5},
            categorical_shift={
                "region": {"east": 0.40, "north": -0.25},
                "channel": {"mobile": 0.50, "web": -0.35},
            },
            concept_rotation=0.6,
        )


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_numeric(n: int, rng: np.random.Generator, spec: DriftSpec) -> pd.DataFrame:
    """Draw the numeric block. Base distributions are fixed; drift shifts them."""
    # Reference location/scale for each numeric feature (mean, std).
    base = {
        "amount": (50.0, 20.0),
        "tenure": (24.0, 12.0),
        "latency": (120.0, 40.0),
        "score": (0.0, 1.0),
        "balance": (1000.0, 500.0),
        "age": (40.0, 12.0),
    }
    cols = {}
    for name, (mu, sigma) in base.items():
        shift = spec.numeric_mean_shift.get(name, 0.0) * sigma
        scale = spec.numeric_scale.get(name, 1.0)
        cols[name] = rng.normal(mu + shift, sigma * scale, size=n)
    return pd.DataFrame(cols)


def _sample_categorical(n: int, rng: np.random.Generator, spec: DriftSpec) -> pd.DataFrame:
    cols = {}
    for name, probs in _CATEGORIES.items():
        levels = list(probs.keys())
        weights = np.array([probs[l] for l in levels], dtype=float)
        # Apply covariate drift on the category probabilities.
        shift = spec.categorical_shift.get(name, {})
        if shift:
            weights = weights + np.array([shift.get(l, 0.0) for l in levels])
            weights = np.clip(weights, 1e-6, None)
        weights = weights / weights.sum()
        cols[name] = rng.choice(levels, size=n, p=weights)
    return pd.DataFrame(cols)


def _make_target(
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    rng: np.random.Generator,
    concept_rotation: float,
) -> np.ndarray:
    """Generate the binary target from a (possibly rotated) logistic model."""
    # Standardize numeric features against the *reference* location/scale so the
    # relationship is stable regardless of covariate drift.
    ref_mu = np.array([50.0, 24.0, 120.0, 0.0, 1000.0, 40.0])
    ref_sd = np.array([20.0, 12.0, 40.0, 1.0, 500.0, 12.0])
    z = (numeric[NUMERIC_FEATURES].to_numpy() - ref_mu) / ref_sd

    coefs = _NUM_COEFS.copy()
    if concept_rotation > 0:
        # Rotate the coefficient vector toward a different direction. This keeps
        # magnitude similar but changes *which* inputs drive the label.
        alt = np.array([-0.6, 0.8, -0.9, 0.4, 0.7, -1.0])
        coefs = (1 - concept_rotation) * coefs + concept_rotation * alt

    logit = _INTERCEPT + z @ coefs
    for name in CATEGORICAL_FEATURES:
        offsets = _CAT_COEFS[name]
        logit = logit + categorical[name].map(offsets).to_numpy()

    prob = 1.0 / (1.0 + np.exp(-logit))
    return (rng.random(len(prob)) < prob).astype(int)


def generate_dataset(
    n_samples: int = 6000,
    seed: int = 42,
    spec: DriftSpec | None = None,
) -> pd.DataFrame:
    """Generate a single labelled batch.

    Parameters
    ----------
    n_samples: rows to generate.
    seed: RNG seed for reproducibility.
    spec: drift specification; ``None`` means no drift (reference distribution).
    """
    spec = spec or DriftSpec.none()
    rng = _rng(seed)
    numeric = _sample_numeric(n_samples, rng, spec)
    categorical = _sample_categorical(n_samples, rng, spec)
    target = _make_target(numeric, categorical, rng, spec.concept_rotation)
    frame = pd.concat([numeric, categorical], axis=1)
    frame["target"] = target
    return frame


def make_reference_and_current(
    n_samples: int = 6000,
    seed: int = 42,
    current_spec: DriftSpec | None = None,
):
    """Convenience: build a clean reference batch and a (drifted) current batch.

    Returns ``(reference_df, current_df)``. The current batch uses a different
    seed so sampling noise differs even when ``current_spec`` is a no-op.
    """
    reference = generate_dataset(n_samples, seed=seed, spec=DriftSpec.none())
    current = generate_dataset(
        n_samples, seed=seed + 1, spec=current_spec or DriftSpec.moderate()
    )
    return reference, current
