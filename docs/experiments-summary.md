# Experiments summary (Phase 3)

Four scripts under `scripts/` produce figures in `figures/` at 300 DPI (see `config.yaml` → `figures`). **Re-run locally** after changing `dataset.seed` or model settings; numbers below are **illustrative** of structure, not fixed constants.

| Script | Figure | Claim (roadmap) |
|--------|--------|------------------|
| `experiment_a_perfect_feature.py` | `exp_a_perfect_feature.png` | Low $n_c$ + $\hat p=1$ is uncertain; Agresti–Coull wide; smoothing pulls Uruguay toward $\bar p$. |
| `experiment_b_smoothing_effect.py` | `exp_b_smoothing_effect.png` | Naïve vs smoothed vs one-hot **country** encoding + XGBoost; train vs test AUC-PR gap. |
| `experiment_c_correlation_trap.py` | `exp_c_correlation_trap.png` | High TE correlation does **not** imply dropping one feature; compare AUC-PR with both vs single TE + numerics. |
| `experiment_d_encoding_comparison.py` | `exp_d_encoding_comparison.png` | Leaky TE (train+test labels in counts) vs OOF TE; train AUC-PR inflation. |

## How to run

From the **repository root**:

```bash
pip install -r requirements.txt
python scripts/check_env.py    # optional: verify imports
python scripts/run_all.py
```

If imports fail, the error is usually `ModuleNotFoundError` (install `requirements.txt` in the same Python you use to run scripts). If `import src` fails, you are not at the repo root.

## Alignment with roadmap expectations

- **Experiment A:** With Uruguay fixed in-train, you should see $n=5$, $k=5$, and a wide Agresti–Coull interval on the plot.
- **Experiment B:** Expect strong **train** AUC-PR and more modest **test** AUC-PR on rare fraud; ranking across encodings can vary with `seed` and XGBoost settings.
- **Experiment C:** The roadmap asks for Pearson $r > 0.7$ between row-wise naïve target encodings. With **15 countries** and **12 merchants**, pooling dilutes row-level correlation; you may see $r$ in the **0.35–0.55** range unless you re-tune `copula_rho`, `merchant_score_z1` / `merchant_score_z2`, and fraud logits. The script prints a note when $r < 0.7$. The **model comparison** (both vs one TE) may show **merchant TE alone** near **both** when the two encodings carry overlapping information — that is itself a valid discussion point for the article.
- **Experiment D:** Uses a **larger** effective `test_size` (≥ 0.35) so pooled test labels have more impact on leaky encoding. The train vs test gap may still be modest if the model underfits; the figure is best read together with the printed metrics.

## Paper exercises (Phase 3)

- **3-A / 3-B / 3-C:** Pencil-and-paper predictions and the “correlation > 0.7” memo — still valuable even if the synthetic run lands slightly below 0.7 after pooling.
