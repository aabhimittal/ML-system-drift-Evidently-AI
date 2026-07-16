"""Lightweight monitoring: append run metrics to a JSON history + plot trends."""

from .dashboard import (
    append_history,
    load_history,
    plot_history,
)

__all__ = ["append_history", "load_history", "plot_history"]
