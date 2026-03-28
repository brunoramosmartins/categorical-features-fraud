"""Binomial confidence intervals (Agresti–Coull)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def agresti_coull_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Agresti–Coull interval for a Binomial proportion.

    Uses adjusted :math:`\\tilde p = (k+2)/(n+4)` and :math:`\\tilde n = n+4`
    (add-two successes and failures).

    Args:
        k: Observed successes.
        n: Trials. If ``n <= 0``, returns ``(nan, nan)``.
        alpha: Two-sided error rate (default 0.05 → nominal 95% interval).

    Returns:
        ``(lower, upper)`` clipped to ``[0, 1]``.
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
