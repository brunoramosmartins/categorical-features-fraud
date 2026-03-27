"""Experiment A: perfect-feature illusion — low n, wide CI, smoothing collapse."""

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
from src.plotting import apply_plot_style, savefig
from src.stats_utils import agresti_coull_interval


def main() -> None:
    cfg = load_config()
    apply_plot_style(cfg)
    train_df, _ = load_and_split(cfg)
    y_tr = train_df["fraud"]
    p_bar = float(y_tr.mean())

    # Per-country counts and naive TE on train
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

    # Panel A: encoded value vs n_c with error bars (AC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.errorbar(
        stats_df["n_c"],
        stats_df["p_hat"],
        yerr=[stats_df["p_hat"] - stats_df["ci_lo"], stats_df["ci_hi"] - stats_df["p_hat"]],
        fmt="o",
        alpha=0.7,
        capsize=3,
        ecolor="gray",
    )
    uy = stats_df[stats_df["country"] == "Uruguay"]
    if not uy.empty:
        ax.scatter(uy["n_c"], uy["p_hat"], s=120, c="crimson", zorder=5, label="Uruguay")
        ax.annotate(
            "Uruguay",
            (float(uy["n_c"].iloc[0]), float(uy["p_hat"].iloc[0])),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=10,
            color="crimson",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Category sample size $n_c$ (train)")
    ax.set_ylabel("Naïve target encoding $\\hat{p}_c$")
    ax.set_title("A: Encoded rate vs support (95% Agresti–Coull)")
    ax.legend()

    # Panel B: Uruguay smoothed estimate vs prior strength m = alpha+beta
    m_list = list(cfg.get("experiments", {}).get("smoothing_prior_totals", [10, 100, 1000]))
    uy_row = train_df[train_df["country"] == "Uruguay"]
    k_uy = int(uy_row["fraud"].sum())
    n_uy = int(len(uy_row))
    smoothed_vals = []
    for m in m_list:
        alpha = m * p_bar
        beta = m * (1.0 - p_bar)
        # Single-level analytic: (k+alpha)/(n+alpha+beta)
        smoothed_vals.append((k_uy + alpha) / (n_uy + alpha + beta))

    ax2 = axes[1]
    ax2.plot(m_list, smoothed_vals, "o-", color="darkblue", linewidth=2, markersize=8)
    ax2.axhline(p_bar, color="green", linestyle="--", label=f"Global mean $\\bar{{p}}$={p_bar:.4f}")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Prior strength $m=\alpha+\beta$")
    ax2.set_ylabel(r"Smoothed $\tilde{p}$ for Uruguay")
    ax2.set_title("B: Shrinkage toward global mean")
    ax2.legend()

    plt.tight_layout()
    savefig("exp_a_perfect_feature", cfg)
    print("Experiment A: saved figures/exp_a_perfect_feature.png")
    print(f"Uruguay (train): n={n_uy}, k={k_uy}, AC 95% CI example: {agresti_coull_interval(k_uy, n_uy)}")


if __name__ == "__main__":
    main()
