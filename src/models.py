"""XGBoost training and metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from xgboost import XGBClassifier

from src.config_utils import load_config


def build_xgb_classifier(
    cfg: dict[str, Any] | None = None,
    *,
    y_train: np.ndarray | pd.Series | None = None,
) -> XGBClassifier:
    if cfg is None:
        cfg = load_config()
    p = dict(cfg["models"]["xgboost"])
    if y_train is not None:
        y = np.asarray(y_train).astype(int).ravel()
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos > 0:
            p["scale_pos_weight"] = max(1.0, neg / pos)
    return XGBClassifier(**p)


def train_and_evaluate(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Fit XGBoost; return train/test AUC-PR and F1 (threshold 0.5 on proba)."""
    if cfg is None:
        cfg = load_config()
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train).astype(int).ravel()
    y_test = np.asarray(y_test).astype(int).ravel()

    model = build_xgb_classifier(cfg, y_train=y_train)
    model.fit(X_train, y_train, verbose=False)

    proba_tr = model.predict_proba(X_train)[:, 1]
    proba_te = model.predict_proba(X_test)[:, 1]
    pred_tr = (proba_tr >= 0.5).astype(int)
    pred_te = (proba_te >= 0.5).astype(int)

    return {
        "aucpr_train": float(average_precision_score(y_train, proba_tr)),
        "aucpr_test": float(average_precision_score(y_test, proba_te)),
        "f1_train": float(f1_score(y_train, pred_tr, zero_division=0)),
        "f1_test": float(f1_score(y_test, pred_te, zero_division=0)),
    }
