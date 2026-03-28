"""Categorical encodings: one-hot, target (naïve / leaky / smoothed / fold-based)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def _global_mean(y: pd.Series | np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(y.mean()) if len(y) else 0.0


def one_hot_encode(series: pd.Series, feature_name: str) -> pd.DataFrame:
    """Return dummy columns for each observed level (float 0/1).

    Args:
        series: Categorical column to expand.
        feature_name: Prefix for column names ``{feature_name}_{level}``.

    Returns:
        DataFrame with one column per distinct level in ``series``.
    """
    return pd.get_dummies(series.astype(str), prefix=feature_name, dtype=float)


def target_encode_naive(
    X_train: pd.Series,
    y_train: pd.Series,
    X_apply: pd.Series,
    feature_name: str,
    *,
    global_mean: float | None = None,
) -> pd.Series:
    """Naïve target encoding: MLE :math:`k/n` per level on the fit sample.

    Args:
        X_train: Categories used to compute level-wise rates.
        y_train: Binary labels aligned with ``X_train``.
        X_apply: Categories to encode (train, validation, or test).
        feature_name: Base name for the output series.
        global_mean: Imputation for unseen levels; defaults to mean of ``y_train``.

    Returns:
        Series named ``{feature_name}_target_naive`` with one float per row of ``X_apply``.
    """
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
    """Smoothed target encoding: Beta–Binomial posterior mean per level.

    Uses :math:`(k+\\alpha)/(n+\\alpha+\\beta)` with counts :math:`(k,n)` from
    ``(X_train, y_train)`` only.

    Args:
        X_train, y_train: Fit sample for level statistics.
        X_apply: Rows to encode.
        feature_name: Base name for the output series.
        alpha, beta: Beta prior hyperparameters (same for all levels).
        global_mean: Fallback for unseen levels; defaults to mean of ``y_train``.

    Returns:
        Series named ``{feature_name}_target_smooth``.
    """
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
    """Target encoding fit using **concatenated** train and test labels (leakage).

    Intended only for ``experiment_d`` to demonstrate optimistic training metrics
    when test labels influence the mapping.

    Args:
        X_train, y_train, X_test, y_test: Parallel splits; all labels enter the mapping.
        feature_name: Base name for both output series.

    Returns:
        ``(enc_train, enc_test)`` with identical column name suffix ``_target_leaky``.
    """
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
    """Out-of-fold target encoding on training rows (no label leakage within fold).

    Each held-out fold is encoded using statistics from the other folds only.
    If ``alpha == beta == 0``, uses naïve MLE; otherwise smoothed encoding.

    Args:
        X_train, y_train: Full training set to partition with ``KFold``.
        feature_name: Base name for the output series.
        n_folds: Number of CV folds (must be >= 2).
        alpha, beta: Smoothing hyperparameters (0,0 for naïve OOF).
        global_mean: Fill for any remaining NaNs; defaults to mean of ``y_train``.
        random_state: Passed to ``KFold(shuffle=True)``.

    Returns:
        Series aligned with ``X_train``, named ``{feature_name}_target_oof``.
    """
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
    """Encode held-out rows using statistics from ``(X_train, y_train)`` only.

    Pairs with :func:`fold_target_oof` for a proper train/test pipeline: OOF on
    train, full-train mapping on test.

    Args:
        X_train, y_train: Training data defining level rates.
        X_test: Test (or validation) categories.
        feature_name: Base name for the output series.
        alpha, beta: Same semantics as :func:`smoothed_target_encode`; ``(0,0)`` is naïve.
        global_mean: Unseen-level fallback; defaults to mean of ``y_train``.

    Returns:
        Series for ``X_test`` with the same naming convention as naïve/smooth encoders.
    """
    if global_mean is None:
        global_mean = _global_mean(y_train)
    if alpha == 0.0 and beta == 0.0:
        return target_encode_naive(X_train, y_train, X_test, feature_name, global_mean=global_mean)
    return smoothed_target_encode(
        X_train, y_train, X_test, feature_name, alpha, beta, global_mean=global_mean
    )
