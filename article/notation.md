# Notation Glossary

<!-- To be completed in Phase 4 — writing conventions -->

| Symbol | Meaning |
|--------|---------|
| $X$ | Categorical feature |
| $c$ | A specific category level |
| $n_c$ | Number of observations where $X = c$ |
| $k_c$ | Number of positive cases ($Y = 1$) where $X = c$ |
| $\hat{p}_c$ | MLE of $P(Y=1 \mid X=c)$: $k_c / n_c$ |
| $\tilde{p}_c$ | Smoothed (Bayesian posterior) estimate of $P(Y=1 \mid X=c)$ |
| $\alpha, \beta$ | Beta prior hyperparameters |
| $w_c$ | Data weight in smoothed estimate: $n_c / (n_c + \alpha + \beta)$ |
| $\bar{p}$ | Global fraud rate (prior mean) |
| $Y$ | Binary target variable (1 = fraud, 0 = legitimate) |
| $N$ | Total number of observations |
