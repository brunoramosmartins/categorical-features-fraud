# Notation glossary

Conventions align with `article/features-that-lie.md`. Random variables are capitalised where standard; observed counts are integers.

| Symbol | Meaning |
|--------|---------|
| \(X\) | Categorical feature (single column); level \(c\) is one category |
| \(c\) | A specific category level of \(X\) |
| \(Y\) | Binary target (\(1\) = fraud, \(0\) = legitimate) |
| \(n_c\) | Number of rows in the **fit sample** with \(X = c\) |
| \(k_c\) | Number of those rows with \(Y = 1\) |
| \(N\) | Total number of rows in a designated sample (e.g. training set size) |
| \(\hat{p}_c\) | MLE \(k_c / n_c\) of \(P(Y{=}1 \mid X{=}c)\) under Binomial sampling |
| \(\tilde{p}\), \(\tilde{n}\) | Agresti–Coull adjusted proportion \((k{+}2)/(n{+}4)\) and effective \(n{+}4\) |
| \(\alpha, \beta\) | Beta prior hyperparameters; \(m = \alpha + \beta\) is prior strength |
| \(\mu_0\) | Prior mean \(\alpha/(\alpha+\beta)\), often set to global training rate \(\bar{p}\) |
| \(\tilde{p}_c^{\mathrm{Bayes}}\) | Posterior mean \((\alpha + k_c)/(\alpha + \beta + n_c)\) |
| \(w_c\) | Data weight \(n_c / (n_c + \alpha + \beta)\) in the posterior mean decomposition |
| \(\bar{p}\) | Global fraud rate in training (or chosen baseline), e.g. \(\sum_i Y_i / N_{\mathrm{train}}\) |
| \(z_1, z_2\) | Row-wise target-encoded values for two categorical features (§6) |
| OOF | Out-of-fold: statistic for a row computed without that row’s label in the level’s counts |

**Fit scope:** Always state whether \((k_c, n_c)\) are computed on **training only**, **train+test** (leaky), or **out-of-fold** training rows.
