# Experiments summary (Phase 3)

Four scripts under `scripts/` produce figures in `figures/` at 300 DPI (see `config.yaml` → `figures`). **Re-run locally** after changing `dataset.seed` or model settings; numbers below are **illustrative** of structure, not fixed constants.

| Script | Figure | Claim (roadmap) |
|--------|--------|------------------|
| `experiment_a_perfect_feature.py` | `exp_a_perfect_feature.png` | Low $n_c$ + $\hat p=1$ is uncertain; Agresti–Coull wide; smoothing pulls Uruguay toward $\bar p$. |
| `experiment_b_smoothing_effect.py` | `exp_b_smoothing_effect.png` | Naïve vs smoothed vs one-hot **country** encoding + XGBoost; train vs test AUC-PR gap. |
| `experiment_c_correlation_trap.py` | `exp_c_correlation_trap.png` | Row-wise naïve TEs can show **moderate** Pearson $r$ after pooling; mutual information and hold-out AUC-PR still justify keeping **both** columns when the DGP needs both latent dimensions (see `config.yaml` fraud logits). |
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
- **Experiment C:** With **15 countries** and **12 merchants**, pooling typically keeps Pearson $r$ between row-wise naïve TEs in the **0.35–0.45** range even when latent $(z_1,z_2)$ correlation is high (`copula_rho`). The default **fraud** logit emphasises a **strong interaction** in $(z_1,z_2)$ so **both TEs + numerics** can beat **either TE alone** on test AUC-PR — check the printed line `Both:` vs `Merchant only:`. The script reminds you when $r$ is low; treat panels (b)–(c) as the primary quantitative story alongside (a).
- **Experiment D:** Uses a **larger** effective `test_size` (≥ 0.35) so pooled test labels have more impact on leaky encoding. The train vs test gap may still be modest if the model underfits; the figure is best read together with the printed metrics.

## Paper exercises (Phase 3)

- **3-A / 3-B / 3-C:** Pencil-and-paper predictions and the “correlation > 0.7” memo — still valuable even if the synthetic run lands slightly below 0.7 after pooling.
