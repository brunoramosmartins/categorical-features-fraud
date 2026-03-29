"""Experiment A: low support — wide Agresti–Coull intervals; smoothing shrinkage."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_utils import load_config
from src.data import load_and_split
from src.plotting import academic_colors, apply_plot_style, savefig
from src.stats_utils import agresti_coull_interval


def main() -> None:
    cfg = load_config()
    C = academic_colors()
    apply_plot_style(cfg)
    train_df, _ = load_and_split(cfg)
    y_tr = train_df["fraud"]
    p_bar = float(y_tr.mean())

    stats_df = (
        train_df.groupby("country", observed=False)
        .agg(n_c=("fraud", "size"), k_c=("fraud", "sum"))
        .reset_index()
    )
    stats_df["p_hat"] = stats_df["k_c"] / stats_df["n_c"]
    stats_df["ci_lo"] = np.nan
    stats_df["ci_hi"] = np.nan
    for i, row in stats_df.iterrows():
        lo, hi = agresti_coull_interval(int(row["k_c"]), int(row["n_c"]))
        stats_df.at[i, "ci_lo"] = lo
        stats_df.at[i, "ci_hi"] = hi

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    ax = axes[0]
    ax.errorbar(
        stats_df["n_c"],
        stats_df["p_hat"],
        yerr=[stats_df["p_hat"] - stats_df["ci_lo"], stats_df["ci_hi"] - stats_df["p_hat"]],
        fmt="o",
        ms=4,
        color=C["gray"],
        ecolor=C["light"],
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        zorder=2,
        label="Levels (train)",
    )
    uy = stats_df[stats_df["country"] == "Uruguay"]
    if not uy.empty:
        ax.scatter(
            uy["n_c"],
            uy["p_hat"],
            s=70,
            facecolors="white",
            edgecolors=C["accent"],
            linewidths=1.4,
            zorder=5,
            label="Sparse anchor (synthetic)",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Level count $n_c$ (training)")
    ax.set_ylabel(r"Naive target rate $\hat{p}_c$")
    ax.set_title("(a) Rate vs. support with 95% Agresti–Coull intervals")
    ax.set_ylim(-0.02, min(1.05, ax.get_ylim()[1]))
    ax.legend(loc="lower right", fontsize=8)

    m_list = list(cfg.get("experiments", {}).get("smoothing_prior_totals", [10, 100, 1000]))
    uy_row = train_df[train_df["country"] == "Uruguay"]
    k_uy = int(uy_row["fraud"].sum())
    n_uy = int(len(uy_row))
    smoothed_vals = []
    for m in m_list:
        alpha = m * p_bar
        beta = m * (1.0 - p_bar)
        smoothed_vals.append((k_uy + alpha) / (n_uy + alpha + beta))

    ax2 = axes[1]
    ax2.plot(
        m_list,
        smoothed_vals,
        "o-",
        color=C["secondary"],
        linewidth=1.2,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1,
        markeredgecolor=C["secondary"],
    )
    ax2.axhline(p_bar, color=C["gray"], linestyle="--", linewidth=1, label=rf"Global $\bar{{p}}={p_bar:.4f}$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Prior strength $m = \alpha+\beta$")
    ax2.set_ylabel(r"Smoothed rate $\tilde{p}$ (sparse anchor)")
    ax2.set_title("(b) Bayesian shrinkage toward global fraud rate")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    savefig("exp_a_perfect_feature", cfg)
    print("Experiment A: saved figures/exp_a_perfect_feature.png")
    print(f"Sparse anchor (train): n={n_uy}, k={k_uy}, AC 95% CI: {agresti_coull_interval(k_uy, n_uy)}")


if __name__ == "__main__":
    main()
