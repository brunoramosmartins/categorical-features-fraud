"""Categorical encodings: one-hot, target (naïve / leaky / smoothed / fold-based)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def _global_mean(y: pd.Series | np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(y.mean()) if len(y) else 0.0


def one_hot_encode(series: pd.Series, feature_name: str) -> pd.DataFrame:
    """One-hot columns `{feature_name}_{level}`."""
    return pd.get_dummies(series.astype(str), prefix=feature_name, dtype=float)


def target_encode_naive(
    X_train: pd.Series,
    y_train: pd.Series,
    X_apply: pd.Series,
    feature_name: str,
    *,
    global_mean: float | None = None,
) -> pd.Series:
    """MLE k/n on training rows; apply to X_apply; unseen → global_mean."""
    if global_mean is None:
        global_mean = _global_mean(y_train)
    df = pd.DataFrame({"x": X_train, "y": y_train.astype(float)})
    grp = df.groupby("x", observed=False)["y"].agg(["sum", "count"])
    mapping = (grp["sum"] / grp["count"]).to_dict()
    out = X_apply.map(mapping)
    return out.fillna(global_mean).rename(f"{feature_name}_target_naive")


def smoothed_target_encode(
    X_train: pd.Series,
    y_train: pd.Series,
    X_apply: pd.Series,
    feature_name: str,
    alpha: float,
    beta: float,
    *,
    global_mean: float | None = None,
) -> pd.Series:
    """Beta–Binomial posterior mean (alpha+k)/(alpha+beta+n) fit on train only."""
    if global_mean is None:
        global_mean = _global_mean(y_train)
    df = pd.DataFrame({"x": X_train, "y": y_train.astype(float)})
    grp = df.groupby("x", observed=False)["y"].agg(["sum", "count"])
    num = grp["sum"] + alpha
    den = grp["count"] + alpha + beta
    mapping = (num / den).to_dict()
    out = X_apply.map(mapping)
    return out.fillna(global_mean).rename(f"{feature_name}_target_smooth")


def target_encode_leaky(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    feature_name: str,
) -> tuple[pd.Series, pd.Series]:
    """Naïve TE using train+test labels together (deliberate leakage for Exp D)."""
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    g = _global_mean(y_full)
    enc_train_leak = target_encode_naive(X_full, y_full, X_train, feature_name, global_mean=g)
    enc_test_leak = target_encode_naive(X_full, y_full, X_test, feature_name, global_mean=g)
    enc_train_leak.name = f"{feature_name}_target_leaky"
    enc_test_leak.name = f"{feature_name}_target_leaky"
    return enc_train_leak, enc_test_leak


def fold_target_oof(
    X_train: pd.Series,
    y_train: pd.Series,
    feature_name: str,
    n_folds: int,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
    global_mean: float | None = None,
    random_state: int = 0,
) -> pd.Series:
    """Out-of-fold target encoding on each training row (no in-fold label in statistic)."""
    if global_mean is None:
        global_mean = _global_mean(y_train)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True).astype(float)
    oof = np.full(len(X_train), np.nan, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X_arr = X_train.to_numpy()
    y_arr = y_train.to_numpy()
    for tr_idx, ho_idx in kf.split(X_train):
        x_tr = pd.Series(X_arr[tr_idx])
        y_tr = pd.Series(y_arr[tr_idx])
        x_ho = pd.Series(X_arr[ho_idx])
        if alpha == 0.0 and beta == 0.0:
            enc = target_encode_naive(x_tr, y_tr, x_ho, feature_name, global_mean=global_mean)
        else:
            enc = smoothed_target_encode(
                x_tr, y_tr, x_ho, feature_name, alpha, beta, global_mean=global_mean
            )
        oof[ho_idx] = enc.to_numpy()
    return pd.Series(oof, name=f"{feature_name}_target_oof").fillna(global_mean)


def fold_target_apply_test(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    feature_name: str,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
    global_mean: float | None = None,
) -> pd.Series:
    """Encode test rows using full training statistics only (no test labels)."""
    if global_mean is None:
        global_mean = _global_mean(y_train)
    if alpha == 0.0 and beta == 0.0:
        return target_encode_naive(X_train, y_train, X_test, feature_name, global_mean=global_mean)
    return smoothed_target_encode(
        X_train, y_train, X_test, feature_name, alpha, beta, global_mean=global_mean
    )
