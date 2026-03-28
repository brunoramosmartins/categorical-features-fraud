"""Matplotlib/Seaborn style from config and PNG export under ``figures/``."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.config_utils import load_config, repo_root


def apply_plot_style(cfg: dict | None = None) -> None:
    """Apply Seaborn theme and ``figure.figsize`` from ``cfg['figures']``."""
    if cfg is None:
        cfg = load_config()
    fig = cfg.get("figures", {})
    style = fig.get("style", "seaborn-v0_8-whitegrid")
    try:
        sns.set_theme(style=style.replace("seaborn-v0_8-", "") if "seaborn" in style else style)
    except Exception:
        sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = tuple(fig.get("figsize", [10, 6]))


def figures_dir() -> Path:
    """Directory where experiment scripts save PNGs (created on save if missing)."""
    return repo_root() / "figures"


def savefig(name: str, cfg: dict | None = None) -> Path:
    """Save current figure to figures/{name}.png at configured DPI."""
    if cfg is None:
        cfg = load_config()
    dpi = int(cfg.get("figures", {}).get("dpi", 300))
    out_dir = figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path
