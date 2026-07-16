"""Configuration loading.

A thin, dependency-light wrapper around the YAML config file. We expose a
``Config`` object that supports both attribute access (``cfg.drift``) and
dict-style access (``cfg["drift"]``), plus a couple of convenience helpers for
resolving and creating the artifact directories used across the project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Repository root = three levels up from this file (src/ml_drift/config.py).
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "config.yaml"


class Config:
    """Read-only view over the parsed configuration dictionary."""

    def __init__(self, data: Dict[str, Any], root: Path):
        self._data = data
        self.root = root

    # -- access helpers -----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, key: str) -> Any:  # only called if normal lookup fails
        try:
            return self._data[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    # -- path helpers -------------------------------------------------------
    def path(self, dotted_key: str) -> Path:
        """Resolve a ``paths.*`` entry (e.g. ``"model_dir"``) to an absolute path."""
        value = self._data["paths"][dotted_key]
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    def ensure_dirs(self) -> None:
        """Create every directory referenced under ``paths``."""
        for key, value in self._data["paths"].items():
            p = Path(value)
            target = p if p.is_absolute() else (self.root / p)
            # Treat entries ending in a known file extension as files.
            if target.suffix in {".csv", ".json", ".joblib", ".html"}:
                target.parent.mkdir(parents=True, exist_ok=True)
            else:
                target.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load configuration from ``path`` (defaults to ``config/config.yaml``)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    # Root is the directory that contains the `config/` folder.
    root = cfg_path.resolve().parents[1]
    return Config(data, root)
