"""Matplotlib/Seaborn style from config and PNG export under ``figures/``."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.config_utils import load_config, repo_root

# Muted, print-friendly palette (grays + one accent for highlights)
_ACADEMIC_COLORS = {
    "ink": "#1a1a1a",
    "gray": "#555555",
    "light": "#b0b0b0",
    "accent": "#8b1538",
    "secondary": "#1f4e79",
    "tertiary": "#2d5c4a",
    "muted_purple": "#5c4d7d",
    "fill": "#e8e8e8",
}


def academic_colors() -> dict[str, str]:
    """Named colours for experiment scripts (consistent with academic_style)."""
    return dict(_ACADEMIC_COLORS)


def _apply_academic_rcparams(dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif",
                "Bitstream Vera Serif",
                "Computer Modern Roman",
                "Times New Roman",
                "Times",
            ],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.linewidth": 0.45,
            "grid.linestyle": "-",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "0.85",
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": dpi,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
    sns.set_theme(style="ticks", context="paper", rc={"axes.grid": True})


def apply_plot_style(cfg: dict | None = None) -> None:
    """Apply figure style from ``cfg['figures']``: academic (default) or legacy seaborn."""
    if cfg is None:
        cfg = load_config()
    fig = cfg.get("figures", {})
    dpi = int(fig.get("dpi", 300))
    plt.rcParams["figure.figsize"] = tuple(fig.get("figsize", [10, 5.5]))

    if fig.get("academic_style", True):
        _apply_academic_rcparams(dpi)
        return

    style = fig.get("style", "seaborn-v0_8-whitegrid")
    try:
        sns.set_theme(
            style=style.replace("seaborn-v0_8-", "") if "seaborn" in style else style
        )
    except Exception:
        sns.set_theme(style="whitegrid")


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
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    return path
