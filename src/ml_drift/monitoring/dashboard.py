"""Persist and visualise drift/performance metrics over successive runs.

Each pipeline run appends one record to a JSON history file. Over many runs this
becomes a simple time series of drift share, performance and remediation actions
that the Streamlit app (or ``plot_history``) can render — the minimal version of
an Evidently monitoring dashboard, with no external service required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_history(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []


def append_history(path: str | Path, record: Dict[str, Any]) -> Path:
    """Append ``record`` to the JSON history list at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = load_history(path)
    history.append(record)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, default=str)
    return path


def plot_history(path: str | Path, out_path: str | Path) -> Path | None:
    """Render drift-share and performance trends to a PNG. Returns None if empty."""
    history = load_history(path)
    if not history:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional at runtime
        return None

    runs = list(range(1, len(history) + 1))
    drift_share = [r.get("drift_share", 0.0) for r in history]
    champ = [r.get("champion_metric") for r in history]
    chall = [r.get("challenger_metric") for r in history]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(runs, drift_share, alpha=0.3, color="tab:orange", label="drift share")
    ax1.set_xlabel("run")
    ax1.set_ylabel("drift share", color="tab:orange")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(runs, champ, "o-", color="tab:blue", label="champion metric")
    if any(c is not None for c in chall):
        ax2.plot(runs, chall, "s--", color="tab:green", label="challenger metric")
    ax2.set_ylabel("performance", color="tab:blue")

    fig.suptitle("Drift & performance over runs")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
