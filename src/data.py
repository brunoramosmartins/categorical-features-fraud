"""Synthetic dataset generation and train/test split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config_utils import load_config


def load_and_split(
    cfg: dict[str, Any] | None = None,
    *,
    config_path: str | None = None,
    keep_anchor_countries_in_train: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate dataset and return stratified (train, test) DataFrames.

    When ``keep_anchor_countries_in_train`` (default), rows with ``country ==
    \"Uruguay\"`` are always placed in training so low-support narrative
    statistics (e.g. Experiment A) see all ~5 anchor rows in-train.
    """
    if cfg is None:
        cfg = load_config(config_path)
    df = generate_dataset(cfg)
    ds = cfg["dataset"]
    test_size = float(ds["test_size"])
    seed = int(ds["seed"])

    if keep_anchor_countries_in_train:
        uy = df["country"] == "Uruguay"
        uy_df = df[uy].reset_index(drop=True)
        rest = df[~uy].reset_index(drop=True)
        train_df, test_df = train_test_split(
            rest,
            test_size=test_size,
            random_state=seed,
            stratify=rest["fraud"],
        )
        train_df = pd.concat([uy_df, train_df], ignore_index=True)
        train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df["fraud"],
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
    return train_df, test_df


def generate_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    """Build synthetic fraud data: Gaussian copula + stratified categories (Phase 3)."""
    ds = cfg["dataset"]
    mode = str(ds.get("generation", "copula")).lower()
    if mode == "copula":
        return _generate_copula(cfg)
    return _generate_legacy_coupling(cfg)


def _generate_copula(cfg: dict[str, Any]) -> pd.DataFrame:
    """Correlated latent (z1,z2) → country (z1), merchant (mix of z1,z2), fraud (nonlinear)."""
    ds = cfg["dataset"]
    rng = np.random.default_rng(int(ds["seed"]))
    n = int(ds["size"])
    rho = float(ds.get("copula_rho", 0.88))

    country_params = ds["country_params"]
    merchants_cfg = ds["merchant_categories"]
    device_cfg = ds["device_os"]
    channel_cfg = ds["channel"]

    country_names = list(country_params.keys())
    assert "Uruguay" in country_params

    uy = "Uruguay"
    n_uy = int(country_params[uy][0])
    others = [c for c in country_names if c != uy]
    w_other = np.array([country_params[c][0] for c in others], dtype=float)
    w_other = w_other / w_other.sum()
    counts_rest = rng.multinomial(n - n_uy, w_other)
    count_map = {uy: n_uy}
    for c, k in zip(others, counts_rest):
        count_map[c] = int(k)

    merchant_names = list(merchants_cfg.keys())
    m_weight = np.array([merchants_cfg[m][0] for m in merchant_names], dtype=float)
    m_weight = m_weight / m_weight.sum()

    dev_names = list(device_cfg.keys())
    d_weight = np.array([device_cfg[d][0] for d in dev_names], dtype=float)
    d_weight = d_weight / d_weight.sum()

    ch_names = list(channel_cfg.keys())
    ch_weight = np.array([channel_cfg[c][0] for c in ch_names], dtype=float)
    ch_weight = ch_weight / ch_weight.sum()

    Z = rng.multivariate_normal(np.zeros(2), np.array([[1.0, rho], [rho, 1.0]]), size=n)
    z1, z2 = Z[:, 0], Z[:, 1]

    # Country: low z1 → rare countries (Uruguay first)
    order_c = np.argsort(z1)
    country_arr = np.empty(n, dtype=object)
    pos = 0
    for country, cnt in sorted(count_map.items(), key=lambda x: x[1]):
        country_arr[order_c[pos : pos + cnt]] = country
        pos += cnt

    # Merchant: sorted by mix correlated with z1,z2 → high TE correlation with country TE
    score_m = float(ds.get("merchant_score_z1", 0.58)) * z1 + float(ds.get("merchant_score_z2", 0.42)) * z2
    order_m = np.argsort(score_m)
    m_counts = rng.multinomial(n, m_weight)
    merchant_arr = np.empty(n, dtype=object)
    pos = 0
    for mname, cnt in zip(merchant_names, m_counts):
        merchant_arr[order_m[pos : pos + cnt]] = mname
        pos += cnt

    device_idx = rng.choice(len(dev_names), size=n, p=d_weight)
    device_arr = np.array(dev_names)[device_idx]

    ch_idx = rng.choice(len(ch_names), size=n, p=ch_weight)
    channel_arr = np.array(ch_names)[ch_idx]

    fraud = np.zeros(n, dtype=np.int32)
    uy_mask = country_arr == uy
    fraud[uy_mask] = 1

    z_logit = float(ds.get("fraud_intercept", -4.55))
    z_logit = z_logit + float(ds.get("fraud_coef_z1", 2.05)) * z1
    z_logit = z_logit + float(ds.get("fraud_coef_z2", 1.65)) * z2
    z_logit = z_logit + float(ds.get("fraud_coef_interaction", 0.48)) * z1 * z2
    z_logit = z_logit + rng.normal(0.0, float(ds.get("fraud_logit_noise", 0.38)), size=n)

    p = 1.0 / (1.0 + np.exp(-np.clip(z_logit, -30, 30)))
    non_uy = ~uy_mask
    fraud[non_uy] = rng.binomial(1, p[non_uy])

    df = pd.DataFrame(
        {
            "country": country_arr,
            "merchant_category": merchant_arr,
            "device_os": device_arr,
            "channel": channel_arr,
            "fraud": fraud,
        }
    )

    df = df.sample(frac=1.0, random_state=int(ds["seed"])).reset_index(drop=True)

    n_rows = len(df)
    rng2 = np.random.default_rng(int(ds["seed"]) + 17)
    base = rng2.normal(0, 1, size=(n_rows, 5))
    base[:, 0] += df["fraud"].to_numpy(dtype=float) * 0.2
    base[:, 1] += df["fraud"].to_numpy(dtype=float) * 0.12
    num_cols = ds.get(
        "numerical_features",
        [
            "transaction_amount",
            "account_age_days",
            "transaction_hour",
            "distance_from_home",
            "num_transactions_last_24h",
        ],
    )
    for j, col in enumerate(num_cols):
        df[col] = base[:, j]

    return df


def _generate_legacy_coupling(cfg: dict[str, Any]) -> pd.DataFrame:
    """Fallback: sequential P(merchant|country) coupling (weaker TE correlation)."""
    ds = cfg["dataset"]
    rng = np.random.default_rng(int(ds["seed"]))
    n = int(ds["size"])
    country_params = ds["country_params"]
    merchants_cfg = ds["merchant_categories"]
    device_cfg = ds["device_os"]
    channel_cfg = ds["channel"]

    country_names = list(country_params.keys())
    uy = "Uruguay"
    n_uy = int(country_params[uy][0])
    others = [c for c in country_names if c != uy]
    w_other = np.array([country_params[c][0] for c in others], dtype=float)
    w_other = w_other / w_other.sum()
    counts_rest = rng.multinomial(n - n_uy, w_other)
    count_map = {uy: n_uy}
    for c, k in zip(others, counts_rest):
        count_map[c] = int(k)

    fraud_rate_country = {c: float(country_params[c][1]) for c in country_names}
    merchant_names = list(merchants_cfg.keys())
    m_weight = np.array([merchants_cfg[m][0] for m in merchant_names], dtype=float)
    m_weight = m_weight / m_weight.sum()
    fraud_rate_merchant = np.array([float(merchants_cfg[m][1]) for m in merchant_names])

    dev_names = list(device_cfg.keys())
    d_weight = np.array([device_cfg[d][0] for d in dev_names], dtype=float)
    d_weight = d_weight / d_weight.sum()
    fraud_rate_device = {d: float(device_cfg[d][1]) for d in dev_names}

    ch_names = list(channel_cfg.keys())
    ch_weight = np.array([channel_cfg[c][0] for c in ch_names], dtype=float)
    ch_weight = ch_weight / ch_weight.sum()
    fraud_rate_channel = {c: float(channel_cfg[c][1]) for c in ch_names}

    k_couple = float(ds.get("merchant_country_coupling", 12.0))
    stick = float(ds.get("merchant_country_stickiness", 1.0))
    preferred_merchant = {c: merchant_names[i % len(merchant_names)] for i, c in enumerate(country_names)}

    rows: list[dict[str, Any]] = []
    for country, cnt in count_map.items():
        fr_c = fraud_rate_country[country]
        for _ in range(cnt):
            if rng.random() < stick:
                merchant = preferred_merchant[country]
            else:
                w_m = m_weight * (1.0 + k_couple * fr_c * fraud_rate_merchant)
                w_m = w_m / w_m.sum()
                merchant = merchant_names[int(rng.choice(len(merchant_names), p=w_m))]

            device_os = dev_names[int(rng.choice(len(dev_names), p=d_weight))]
            channel = ch_names[int(rng.choice(len(ch_names), p=ch_weight))]

            if country == uy:
                fraud = 1
            else:
                fr_m = float(merchants_cfg[merchant][1])
                z = -6.2 + 3.0 * fr_c + 2.5 * fr_m + 0.8 * fraud_rate_device[device_os]
                z += 0.5 * fraud_rate_channel[channel] + rng.normal(0.0, 0.35)
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                fraud = int(rng.binomial(1, p))

            rows.append(
                {
                    "country": country,
                    "merchant_category": merchant,
                    "device_os": device_os,
                    "channel": channel,
                    "fraud": fraud,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=int(ds["seed"])).reset_index(drop=True)
    n_rows = len(df)
    base = rng.normal(0, 1, size=(n_rows, 5))
    base += df["fraud"].to_numpy(dtype=float)[:, None] * 0.15
    num_cols = ds.get(
        "numerical_features",
        [
            "transaction_amount",
            "account_age_days",
            "transaction_hour",
            "distance_from_home",
            "num_transactions_last_24h",
        ],
    )
    for j, col in enumerate(num_cols):
        df[col] = base[:, j]
    return df
