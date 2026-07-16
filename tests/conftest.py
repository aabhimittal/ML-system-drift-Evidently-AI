"""Shared pytest fixtures."""
import sys
from pathlib import Path

import pytest

# Ensure `src` is importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_drift.data.generate import DriftSpec, generate_dataset  # noqa: E402
from ml_drift.models.train import train_model  # noqa: E402


@pytest.fixture(scope="session")
def reference_df():
    return generate_dataset(2000, seed=1, spec=DriftSpec.none())


@pytest.fixture(scope="session")
def current_moderate_df():
    return generate_dataset(2000, seed=2, spec=DriftSpec.moderate())


@pytest.fixture(scope="session")
def current_none_df():
    return generate_dataset(2000, seed=3, spec=DriftSpec.none())


@pytest.fixture(scope="session")
def current_severe_df():
    return generate_dataset(2000, seed=4, spec=DriftSpec.severe())


@pytest.fixture(scope="session")
def champion(reference_df):
    return train_model(reference_df, model_type="random_forest",
                       params={"n_estimators": 80, "max_depth": 8})
