"""Binomial confidence intervals (Agresti–Coull)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def agresti_coull_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Two-sided Agresti–Coull interval for Binomial proportion.
    Uses adjusted p̃ = (k+2)/(n+4) and ñ = n+4 (add-two successes/failures).
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p_tilde = (k + 2) / (n + 4)
    n_tilde = n + 4
    margin = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    lo = max(0.0, p_tilde - margin)
    hi = min(1.0, p_tilde + margin)
    return (float(lo), float(hi))
