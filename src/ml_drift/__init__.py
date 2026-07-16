"""ml_drift: end-to-end ML data-drift detection and post-detection optimization.

Public sub-packages
-------------------
- ``ml_drift.data``          synthetic data generation + drift injection
- ``ml_drift.models``        baseline model training / prediction
- ``ml_drift.drift``         native drift detection + Evidently reports
- ``ml_drift.optimization``  post-detection remediation (retrain / reweight / stabilize)
- ``ml_drift.pipeline``      end-to-end orchestration
- ``ml_drift.monitoring``    metric history logging
"""

from .config import Config, load_config

__version__ = "0.1.0"

__all__ = ["Config", "load_config", "__version__"]
