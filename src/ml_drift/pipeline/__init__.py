"""End-to-end orchestration of the drift + optimization system."""

from .orchestrator import PipelineResult, run_pipeline

__all__ = ["PipelineResult", "run_pipeline"]
