"""Experiment C: target-encoded features can correlate yet both inform Y."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_utils import load_config
from src.data import load_and_split
from src.encoding import target_encode_naive
from src.models import train_and_evaluate
from src.plotting import academic_colors, apply_plot_style, savefig


def _nums(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    return df[cfg["dataset"]["numerical_features"]].to_numpy(dtype=np.float32)


def main() -> None:
    cfg = load_config()
    C = academic_colors()
    apply_plot_style(cfg)
    train_df, test_df = load_and_split(cfg)
    y_tr, y_te = train_df["fraud"], test_df["fraud"]
    p_bar = float(y_tr.mean())

    zc_tr = target_encode_naive(
        train_df["country"], y_tr, train_df["country"], "country", global_mean=p_bar
    )
    zm_tr = target_encode_naive(
        train_df["merchant_category"],
        y_tr,
        train_df["merchant_category"],
        "merchant_category",
        global_mean=p_bar,
    )
    zc_te = target_encode_naive(train_df["country"], y_tr, test_df["country"], "country", global_mean=p_bar)
    zm_te = target_encode_naive(
        train_df["merchant_category"],
        y_tr,
        test_df["merchant_category"],
        "merchant_category",
        global_mean=p_bar,
    )

    r = float(np.corrcoef(zc_tr.to_numpy(), zm_tr.to_numpy())[0, 1])
    print(f"Pearson corr(TE_country, TE_merchant) on train: {r:.3f}")
    if r < 0.55:
        print(
            "Note: with many pooled levels, row-wise naive TE correlation is often moderate "
            "even when latent (z1,z2) correlation is high; see docs/dataset-design.md."
        )

    mi_c = float(
        mutual_info_classif(zc_tr.to_numpy().reshape(-1, 1), y_tr, random_state=0)[0]
    )
    mi_m = float(
        mutual_info_classif(zm_tr.to_numpy().reshape(-1, 1), y_tr, random_state=0)[0]
    )
    print(f"MI(country TE; Y)={mi_c:.5f}, MI(merchant TE; Y)={mi_m:.5f}")

    Ntr = _nums(train_df, cfg)
    Nte = _nums(test_df, cfg)

    Xb_tr = np.c_[Ntr, zc_tr.to_numpy(), zm_tr.to_numpy()]
    Xb_te = np.c_[Nte, zc_te.to_numpy(), zm_te.to_numpy()]
    Xc_tr, Xc_te = np.c_[Ntr, zc_tr.to_numpy()], np.c_[Nte, zc_te.to_numpy()]
    Xm_tr, Xm_te = np.c_[Ntr, zm_tr.to_numpy()], np.c_[Nte, zm_te.to_numpy()]

    mb = train_and_evaluate(Xb_tr, y_tr, Xb_te, y_te, cfg)
    mc = train_and_evaluate(Xc_tr, y_tr, Xc_te, y_te, cfg)
    mm = train_and_evaluate(Xm_tr, y_tr, Xm_te, y_te, cfg)

    print(f"Both:  test AUC-PR={mb['aucpr_test']:.4f}")
    print(f"Country only: test AUC-PR={mc['aucpr_test']:.4f}")
    print(f"Merchant only: test AUC-PR={mm['aucpr_test']:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    axes[0].scatter(
        zc_tr,
        zm_tr,
        alpha=0.12,
        s=4,
        c=C["secondary"],
        edgecolors="none",
        rasterized=True,
    )
    axes[0].set_xlabel(r"Target encoding: country ($z_1$)")
    axes[0].set_ylabel(r"Target encoding: merchant ($z_2$)")
    axes[0].set_title(f"(a) Row-wise naive target encodings (Pearson $r={r:.2f}$)")

    models = ["Both\nTE", "Country\nTE only", "Merchant\nTE only"]
    scores = [mb["aucpr_test"], mc["aucpr_test"], mm["aucpr_test"]]
    cols_bar = [C["secondary"], C["gray"], C["gray"]]
    axes[1].bar(
        models,
        scores,
        color=cols_bar,
        edgecolor=C["ink"],
        linewidth=0.6,
    )
    axes[1].set_ylabel("AUC-PR (test)")
    axes[1].set_title("(b) Model comparison on hold-out")
    axes[1].set_ylim(0, max(scores) * 1.12 + 1e-6)

    mi_vals = [mi_c, mi_m]
    axes[2].bar(
        ["MI(country TE; Y)", "MI(merchant TE; Y)"],
        mi_vals,
        color=[C["secondary"], C["muted_purple"]],
        edgecolor=C["ink"],
        linewidth=0.6,
    )
    axes[2].set_ylabel("Mutual information (nats)")
    axes[2].set_title("(c) Each encoding carries target information")
    axes[2].tick_params(axis="x", rotation=12)

    plt.tight_layout()
    savefig("exp_c_correlation_trap", cfg)
    print("Experiment C: saved figures/exp_c_correlation_trap.png")


if __name__ == "__main__":
    main()
