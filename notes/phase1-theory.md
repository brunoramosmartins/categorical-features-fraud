# Phase 1 — Statistical foundation (study notes)

Study notes for *estimators under low support* and *correlation ≠ redundancy*. Aligned with the project roadmap (Phase 1).

---

## 1. MLE for category-level fraud rate (Block A)

### Setup

- Categorical feature $X$ with level $c$.
- Among $n_c$ rows with $X = c$, exactly $k_c$ have $Y = 1$ (fraud).
- Treat rows in level $c$ as $k_c$ successes in $n_c$ Bernoulli trials with unknown probability $p_c = P(Y=1 \mid X=c)$.

### Estimator

The sample proportion

$$
\hat{p}_c = \frac{k_c}{n_c}
$$

is the **maximum likelihood estimator** of $p_c$ under a $\mathrm{Binomial}(n_c, p_c)$ likelihood (equivalently, i.i.d. Bernoulli$(p_c)$ for the $n_c$ rows).

### Variance

For each trial $\mathrm{Var}(Y_i) = p_c(1-p_c)$, and for the sum $k_c$ we get $\mathrm{Var}(k_c) = n_c p_c(1-p_c)$. Hence

$$
\mathrm{Var}(\hat{p}_c) = \mathrm{Var}\left(\frac{k_c}{n_c}\right) = \frac{p_c(1-p_c)}{n_c}.
$$

**Low-support regime:** when $n_c$ is small, this variance is large: the estimator is a **high-variance estimator under low support** — the observed $\hat{p}_c$ can look extreme even when $p_c$ is moderate.

### Plug-in variance failure at $\hat{p}_c \in \{0,1\}$

The **plug-in** variance estimate $\widehat{\mathrm{Var}}(\hat{p}_c) = \hat{p}_c(1-\hat{p}_c)/n_c$ is **zero** when $\hat{p}_c = 0$ or $\hat{p}_c = 1$, e.g. $k_c = n_c = 5$. That does *not* mean uncertainty vanished; it is a **failure mode of the Wald interval** for extreme proportions. Better intervals (Wilson, Agresti–Coull, Clopper–Pearson, or Bayesian) remain wide.

---

## 2. Uruguay vs Brazil (numerical deconstruction)

### Uruguay (narrative anchor)

Take $k_c = 5$, $n_c = 5$, so $\hat{p}_c = 1.0$. **Micro-quantification:** in the case dataset, Uruguay had approximately **five** transactions — five.

### Agresti–Coull adjusted proportion

Define

$$
\tilde{p} = \frac{k + 2}{n + 4}, \quad \tilde{n} = n + 4.
$$

For $(k,n) = (5,5)$: $\tilde{p} = 7/9 \approx 0.778$, $\tilde{n} = 9$.

A standard Agresti–Coull **approximate** 95% interval uses

$$
\tilde{p} \pm z_{0.975}\sqrt{\frac{\tilde{p}(1-\tilde{p})}{\tilde{n}}},
$$

with $z_{0.975} \approx 1.96$. Then

$$
\sqrt{\frac{\tilde{p}(1-\tilde{p})}{\tilde{n}}} = \sqrt{\frac{(7/9)(2/9)}{9}} = \frac{\sqrt{14}}{27} \approx 0.1387,
$$

margin $\approx 1.96 \times 0.1387 \approx 0.272$. So

$$
\text{approx. } 95\%\text{ CI} \approx [0.506,\, 1.000]
$$

(clipping the upper endpoint at 1 if needed). The essential point for the article: the interval is **very wide** — a “100% fraud rate” is **compatible with a true rate far below 1**.

**Punchline (statistical wording):** the observed perfection is **more likely an artefact of small $n$** than evidence of a **robust** fraud signal at population level.

### Brazil (contrast)

Illustrative: $n_c \approx 40\,000$, $\hat{p}_c \approx 0.004$. Then

$$
\mathrm{Var}(\hat{p}_c) \approx \frac{0.004 \times 0.996}{40\,000} \approx 9.96\times 10^{-8}, \quad \mathrm{SE}(\hat{p}_c) \approx 3.15\times 10^{-4}.
$$

A 95% Wald-type margin is about $1.96 \times 3.15\times 10^{-4} \approx 6.2\times 10^{-4}$: CI width $\approx 0.0012$ — **tight**, because $n_c$ is large.

### Formal comparison table

| Country (level $c$) | $n_c$ | $k_c$ | $\hat{p}_c$ | Regime | Comment |
|---------------------|-------|-------|-------------|--------|---------|
| Uruguay | $\approx 5$ | $\approx 5$ | $1.0$ | Low support | $\mathrm{Var}(\hat{p}_c)$ large; plug-in variance 0; AC/Wilson intervals wide |
| Brazil | $\approx 40\,000$ | $\approx 160$ (if $\hat{p}\approx 0.004$) | $\approx 0.004$ | High support | Interval width $\sim 10^{-3}$; estimate informative |

*(Exact $k_c$ for Brazil follows from your synthetic generator; the qualitative contrast is what matters for Phase 1.)*

---

## 3. Bayesian smoothing: Beta–Binomial

### Model

- Prior: $p_c \sim \mathrm{Beta}(\alpha, \beta)$.
- Likelihood: $k_c \mid p_c \sim \mathrm{Binomial}(n_c, p_c)$.

### Posterior

Beta is conjugate:

$$
p_c \mid k_c, n_c \sim \mathrm{Beta}(\alpha + k_c,\; \beta + n_c - k_c).
$$

### Posterior mean (smoothed estimate)

$$
\tilde{p}_c^{\mathrm{Bayes}} = \mathbb{E}[p_c \mid k_c, n_c] = \frac{\alpha + k_c}{\alpha + \beta + n_c}.
$$

### Weighted average form

Let prior mean $\mu_0 = \frac{\alpha}{\alpha + \beta}$ and MLE $\hat{p}_c = k_c/n_c$. Write total prior “pseudo-count” $m = \alpha + \beta$. Then

$$
\tilde{p}_c^{\mathrm{Bayes}} = \frac{k_c + \alpha}{n_c + m} = w_c \hat{p}_c + (1 - w_c)\mu_0,
$$

with **data weight**

$$
w_c = \frac{n_c}{n_c + \alpha + \beta} = \frac{n_c}{n_c + m}.
$$

When $n_c$ is small, $w_c$ is small → estimate shrinks toward $\mu_0$ (e.g. global fraud rate). When $n_c$ is large, $w_c \to 1$ → estimate $\approx \hat{p}_c$ (Uruguay vs Brazil story).

### Uruguay vs Brazil (qualitative)

- Uruguay ($n_c = 5$, $k_c = 5$): posterior mean pulls $\hat{p}_c = 1$ toward $\mu_0$.
- Brazil ($n_c$ huge): $w_c \approx 1$, posterior mean $\approx \hat{p}_c$.

### Link to `category_encoders`

Many implementations use smoothing parameters equivalent to adding pseudo-counts $(\alpha, \beta)$ or a single effective prior strength $m$ toward the global mean — same structure as the Beta–Binomial posterior mean.

---

## 4. Correlation $\neq$ redundancy: worked counterexample

### Narrative anchor

A correlation matrix between **target-encoded** features can show $|r| > 0.7$, yet **each feature carries non-redundant information** about $Y$ in the sense that a model using **both** can outperform models using **either alone**. **Marginal correlation between features is not conditional redundancy with respect to $Y$.**

### Toy dataset (8 rows)

Use categorical levels $X_1 \in \{A,B,C\}$, $X_2 \in \{M,N,P\}$, binary $Y$:

| $i$ | $X_1$ | $X_2$ | $Y$ |
|-----|-------|-------|-----|
| 1 | C | M | 0 |
| 2 | A | P | 1 |
| 3 | C | M | 0 |
| 4 | B | P | 1 |
| 5 | B | M | 0 |
| 6 | B | M | 1 |
| 7 | B | P | 0 |
| 8 | C | N | 0 |

### Naïve target encoding (full-sample MLE per level)

$\hat{p}(Y{=}1 \mid X_1{=}A) = 1$, $(B)=1/2$, $(C)=0$.

$\hat{p}(Y{=}1 \mid X_2{=}M) = 1/4$, $(N)=0$, $(P)=2/3$.

Row-wise encoded values $(z_{1,i}, z_{2,i})$:

| $i$ | $z_{1,i}$ | $z_{2,i}$ |
|-----|-----------|-----------|
| 1 | 0 | 0.25 |
| 2 | 1 | 0.666… |
| 3 | 0 | 0.25 |
| 4 | 0.5 | 0.666… |
| 5 | 0.5 | 0.25 |
| 6 | 0.5 | 0.25 |
| 7 | 0.5 | 0.666… |
| 8 | 0 | 0 |

Pearson correlation $\mathrm{Corr}(z_1, z_2) \approx 0.724 > 0.7$.

### Conditional rates differ (non-redundancy at the distribution level)

Examples:

- $P(Y{=}1 \mid X_1{=}A) = 1$ but $P(Y{=}1 \mid X_2{=}M) = 0.25$.
- $P(Y{=}1 \mid X_1{=}B) = 0.5$ vs $P(Y{=}1 \mid X_2{=}P) \approx 0.667$.

So **no** statement of the form “high correlation ⇒ same $P(Y\mid\cdot)$” holds.

### Simple model comparison (illustration)

Using **logistic regression** on the 8 rows with features $(z_1)$ only, $(z_2)$ only, and $(z_1, z_2)$ (sklearn defaults, same regularization):

- Accuracy with $z_1$ only: **0.625**
- Accuracy with $z_2$ only: **0.625**
- Accuracy with both: **0.750**

So **both** encodings together strictly improve over either alone on this toy set — despite **high Pearson correlation** between the two encoded columns across rows.

**Conclusion:** dropping a feature solely because it is correlated with another **encoded** feature is **not** justified without target-aware evidence.

---

## 5. Feature selection under encoding (decision heuristic)

### Methods

| Method | Role | Verdict for target-encoded features |
|--------|------|-------------------------------------|
| Pairwise correlation threshold (e.g. $\|r\| > 0.9$ → drop one) | Between-feature | **Unsafe** — counterexample above |
| Mutual information $I(X;Y)$ | Target-related | **Safe** as a screen (still not a full causal story) |
| Permutation importance (model-based) | Effect on $Y$ via model | **Safe** for “does this help prediction?” |
| SHAP / model explanations | Local/global contribution | **Useful** alongside MI / permutation |

### Five-step heuristic (post–correlation matrix)

1. Compute pairwise correlations between **encoded** features; flag high $|r|$.
2. For each flagged pair, compute **MI with $Y$** (or permutation importance) for **each** feature.
3. If **both** have substantial signal → **keep both** unless a simpler model is required for other reasons — document why.
4. If one has **near-zero** MI / importance → candidate to **drop** (noise or proxy), document evidence.
5. **Document** the decision (numbers + method), not only “we dropped collinear features.”

This is what was **missing** when the case stopped at “we saw high correlation”: a bridge from **observation** to **action** grounded in **relationship to $Y$**.

---

## 6. References (Phase 1)

- Casella & Berger — MLE, variance, large-sample behaviour.
- Agresti & Coull (1998); Brown, Cai & DasGupta (2001) — binomial CIs.
- Bishop (2006); Murphy (2012) — Beta–Binomial / conjugacy.

---

## 7. Paper exercises (pencil and paper)

From the roadmap — **do these by hand** (calculator ok); they are **not** replaced by the Markdown above.

**Exercise 1-A — Confidence interval drill**  
Compute the **95% Agresti–Coull** interval for the fraud rate for each $(k,n)$: $(5,5)$, $(3,10)$, $(1,100)$, $(8,1000)$, $(32,40000)$. Sketch how interval width changes with $n$. At what $n$ does the interval feel “tight enough” for you?

**Exercise 1-B — Smoothing derivation**  
From a $\mathrm{Beta}(\alpha,\beta)$ prior and $\mathrm{Binomial}(n_c,p_c)$ likelihood, derive the posterior, show it is Beta, compute the posterior mean, rewrite it as $w_c\hat{p}_c + (1-w_c)\mu_0$, and plug in Uruguay numbers $(k,n)=(5,5)$ for a chosen $(\alpha,\beta)$.

**Exercise 1-C — Weight function**  
On paper, sketch $w_c = n_c/(n_c+\alpha+\beta)$ vs $n_c$ for $\alpha+\beta \in \{10, 100, 1000\}$ (e.g. $n_c$ from 1 to $10^4$ on log scale). For each prior strength, find roughly where $w_c > 0.9$.

**Exercise 1-D — Counterexample**  
Design your **own** small table (≥8 rows, two categoricals, binary $Y$) with target-encoded correlation $|r|>0.7$ but where you argue both features should be kept — or verify with a tiny model comparison as in §4 above.
