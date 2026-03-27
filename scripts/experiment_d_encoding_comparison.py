"""Experiment D: leaky target encoding vs fold-based (train-only scope)."""

from __future__ import annotations

import copy
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
from src.encoding import (
    fold_target_apply_test,
    fold_target_oof,
    target_encode_leaky,
)
from src.models import train_and_evaluate
from src.plotting import apply_plot_style, savefig


def _nums(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    return df[cfg["dataset"]["numerical_features"]].to_numpy(dtype=np.float32)


def main() -> None:
    cfg = load_config()
    cfg_d = copy.deepcopy(cfg)
    cfg_d["dataset"]["test_size"] = max(float(cfg["dataset"]["test_size"]), 0.35)
    apply_plot_style(cfg)
    train_df, test_df = load_and_split(cfg_d)
    y_tr, y_te = train_df["fraud"], test_df["fraud"]
    n_folds = int(cfg["encoding"]["fold_encoding"]["n_folds"])
    seed = int(cfg["dataset"]["seed"])

    Ntr = _nums(train_df, cfg)
    Nte = _nums(test_df, cfg)

    leak_tr, leak_te = target_encode_leaky(
        train_df["country"], y_tr, test_df["country"], y_te, "country"
    )
    Xl_tr = np.c_[Ntr, leak_tr.to_numpy()]
    Xl_te = np.c_[Nte, leak_te.to_numpy()]
    ml = train_and_evaluate(Xl_tr, y_tr, Xl_te, y_te, cfg)

    oof_tr = fold_target_oof(
        train_df["country"], y_tr, "country", n_folds, random_state=seed
    )
    prop_te = fold_target_apply_test(
        train_df["country"], y_tr, test_df["country"], "country"
    )
    Xp_tr = np.c_[Ntr, oof_tr.to_numpy()]
    Xp_te = np.c_[Nte, prop_te.to_numpy()]
    mp = train_and_evaluate(Xp_tr, y_tr, Xp_te, y_te, cfg)

    print("Leaky:  train AUC-PR=%.4f test AUC-PR=%.4f" % (ml["aucpr_train"], ml["aucpr_test"]))
    print("Proper: train AUC-PR=%.4f test AUC-PR=%.4f" % (mp["aucpr_train"], mp["aucpr_test"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Leaky TE", "Proper (OOF)"]
    tr = [ml["aucpr_train"], mp["aucpr_train"]]
    te = [ml["aucpr_test"], mp["aucpr_test"]]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, tr, w, label="Train", color="indianred")
    ax.bar(x + w / 2, te, w, label="Test", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC-PR")
    ax.set_title("D: Leakage inflates train AUC-PR")
    ax.legend()
    ax.annotate(
        "train vs test gap",
        xy=(0, max(tr[0], te[0])),
        xytext=(0.15, max(tr[0], te[0]) * 0.85),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    plt.tight_layout()
    savefig("exp_d_encoding_comparison", cfg)
    print("Experiment D: saved figures/exp_d_encoding_comparison.png")


if __name__ == "__main__":
    main()
