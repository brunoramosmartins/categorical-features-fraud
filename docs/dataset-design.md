# Synthetic dataset design (implementation)

This file describes the **implemented** generator in `src/data.py` (Phase 3). Parameters live in `config.yaml` under `dataset:`.

## Size and split

- **Rows:** `dataset.size` (default 100,000).
- **Train/test:** `dataset.test_size` (default 0.2), stratified on `fraud`.
- **Uruguay anchor:** all rows with `country == "Uruguay"` are forced into **training** (`load_and_split(..., keep_anchor_countries_in_train=True)`) so Experiment A always sees the full low-support slice (~5 rows) in-train.

## Generation mode: `generation: copula` (default)

1. Sample bivariate normal $(z_1, z_2)$ with correlation `copula_rho`.
2. **Country:** sort rows by $z_1$; assign country labels in contiguous blocks whose sizes match `country_params` counts (rare countries = low $z_1$ tail). Uruguay is the smallest-count level from config (5 rows), all labelled **fraud**.
3. **Merchant:** sort rows by `merchant_score_z1 * z1 + merchant_score_z2 * z2`; assign merchant labels in blocks sized by multinomial draws from `merchant_categories` relative weights.
4. **Device / channel:** i.i.d. draws from `device_os` and `channel` weight tables.
5. **Fraud (non-Uruguay):** logistic model in $(z_1, z_2)$ with configurable intercept, linear terms, interaction `fraud_coef_interaction * z1 * z2`, and Gaussian noise (`fraud_logit_noise`). Tuned so overall prevalence is a few percent (not the nominal 0.5% in the YAML `fraud_rate` key — that key documents intent; the copula uses explicit logit parameters).

## Numerical features

Five Gaussian columns listed in `dataset.numerical_features`, weakly shifted by `fraud` for a small non-categorical signal.

## Legacy mode

`dataset.generation: legacy` uses the older sequential `P(\text{merchant}\mid\text{country})` loop (`_generate_legacy_coupling`). It is weaker for the correlation-trap experiment; **copula is recommended**.

## Outputs

The generator does **not** write CSV to disk by default (keeps repo clean). Scripts call `load_and_split()` in memory.
