"""Experiment B: naive vs smoothed vs one-hot country encoding + XGBoost."""

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
from src.encoding import one_hot_encode, smoothed_target_encode, target_encode_naive
from src.models import train_and_evaluate
from src.plotting import academic_colors, apply_plot_style, savefig


def _num_matrix(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    cols = cfg["dataset"]["numerical_features"]
    return df[cols].to_numpy(dtype=np.float32)


def main() -> None:
    cfg = load_config()
    C = academic_colors()
    apply_plot_style(cfg)
    train_df, test_df = load_and_split(cfg)
    y_tr, y_te = train_df["fraud"], test_df["fraud"]
    enc_cfg = cfg["encoding"]["smoothing"]
    alpha = float(enc_cfg["alpha"])
    beta = float(enc_cfg["beta"])
    p_bar = float(y_tr.mean())

    nums_tr = _num_matrix(train_df, cfg)
    nums_te = _num_matrix(test_df, cfg)

    variants: list[tuple[str, np.ndarray, np.ndarray]] = []

    z_naive_tr = target_encode_naive(
        train_df["country"], y_tr, train_df["country"], "country", global_mean=p_bar
    )
    z_naive_te = target_encode_naive(
        train_df["country"], y_tr, test_df["country"], "country", global_mean=p_bar
    )
    variants.append(
        (
            "Target (naive)",
            np.c_[nums_tr, z_naive_tr.to_numpy()],
            np.c_[nums_te, z_naive_te.to_numpy()],
        )
    )

    z_sm_tr = smoothed_target_encode(
        train_df["country"],
        y_tr,
        train_df["country"],
        "country",
        alpha,
        beta,
        global_mean=p_bar,
    )
    z_sm_te = smoothed_target_encode(
        train_df["country"], y_tr, test_df["country"], "country", alpha, beta, global_mean=p_bar
    )
    variants.append(
        (
            "Target (smoothed)",
            np.c_[nums_tr, z_sm_tr.to_numpy()],
            np.c_[nums_te, z_sm_te.to_numpy()],
        )
    )

    ohe_tr = one_hot_encode(train_df["country"], "country")
    ohe_te = one_hot_encode(test_df["country"], "country")
    ohe_te = ohe_te.reindex(columns=ohe_tr.columns, fill_value=0.0)
    variants.append(
        (
            "One-hot",
            np.c_[nums_tr, ohe_tr.to_numpy()],
            np.c_[nums_te, ohe_te.to_numpy()],
        )
    )

    names = []
    auc_tr = []
    auc_te = []
    for name, Xtr, Xte in variants:
        m = train_and_evaluate(Xtr, y_tr, Xte, y_te, cfg)
        names.append(name)
        auc_tr.append(m["aucpr_train"])
        auc_te.append(m["aucpr_test"])
        print(f"{name}: AUC-PR train={m['aucpr_train']:.4f} test={m['aucpr_test']:.4f} F1 test={m['f1_test']:.4f}")

    bar_cols = [C["secondary"], C["tertiary"], C["gray"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    x = np.arange(len(names))
    axes[0].bar(
        x,
        auc_te,
        color=bar_cols,
        edgecolor=C["ink"],
        linewidth=0.6,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=12, ha="right")
    axes[0].set_ylabel("AUC-PR")
    axes[0].set_title("(a) Test set performance by encoding")
    axes[0].set_ylim(0, max(auc_te) * 1.15 + 1e-6)

    w = 0.36
    axes[1].bar(
        x - w / 2,
        auc_tr,
        w,
        label="Train",
        color=C["fill"],
        edgecolor=C["ink"],
        linewidth=0.6,
    )
    axes[1].bar(
        x + w / 2,
        auc_te,
        w,
        label="Test",
        color=C["secondary"],
        edgecolor=C["ink"],
        linewidth=0.6,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=12, ha="right")
    axes[1].set_ylabel("AUC-PR")
    axes[1].set_title("(b) Train vs. test gap (encoding variance / overfitting)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    savefig("exp_b_smoothing_effect", cfg)
    print("Experiment B: saved figures/exp_b_smoothing_effect.png")


if __name__ == "__main__":
    main()
