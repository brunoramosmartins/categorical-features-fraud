"""Load `config.yaml` from the repository root (paths, seeds, model and figure settings)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_NAME = "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Parse YAML configuration into a nested dict.

    Args:
        path: Explicit config file. Defaults to ``<repo_root>/config.yaml``.

    Returns:
        Parsed configuration (structure expected by ``src.data``, ``src.encoding``, etc.).
    """
    cfg_path = Path(path) if path is not None else _ROOT / _CONFIG_NAME
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    """Absolute path to the project root (parent of ``src/``)."""
    return _ROOT
