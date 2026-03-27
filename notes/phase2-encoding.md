# Phase 2 — Encoding landscape & leakage (study notes)

Connects **what each encoding estimates** (Phase 1 estimator lens) to **native categorical handling**, **leakage** as a consequence of wrong data for that estimator, and practical matrices for the article. Aligned with roadmap Phase 2.

---

## 1. Unified question

For every map from category → number, ask: **what population quantity does this number estimate, and with what variance / leakage risk?** Phase 1 already characterised $\hat{p}_c = k_c/n_c$ as an MLE with variance $\propto 1/n_c$. Encodings that use $Y$ inherit that story.

---

## 2. Encoding strategies (what they estimate)

Let $X$ be a categorical feature with levels $c \in \{1,\ldots,C\}$, $N$ total rows, $n_c$ rows with $X=c$, $k_c$ of those with $Y=1$.

### 2.1 One-hot encoding

For level $c$, column

$$
X_c^{(\mathrm{OH})} = \mathbb{1}[X = c].
$$

**Estimates:** nothing about $Y$. It is an **indicator of the event** $\{X=c\}$, i.e. a nonparametric sufficient coordinate for that level in linear models.

**Target-dependent?** No. **Leakage:** none by construction (no $Y$ used). **Cardinality:** $C$ columns — costly for large $C$ (sparse high-dimensional design).

### 2.2 Frequency encoding

Replace $X=c$ with

$$
X^{(\mathrm{FE})} = \frac{n_c}{N}
$$

(sometimes $\log n_c$, $\sqrt{n_c}$, or rank; the roadmap uses $n_c/N \approx \hat{P}(X=c)$ in large samples).

**Estimates:** (empirical) **category prevalence** $P(X=c)$ — **not** $P(Y\mid X=c)$.

**Target-dependent?** No. **Leakage:** none. **Cardinality:** one column; works well for high $C$ when rarity itself is predictive.

### 2.3 Target encoding (naïve)

Replace $X=c$ with

$$
X^{(\mathrm{TE})} = \hat{p}_c = \frac{k_c}{n_c}.
$$

**Estimates:** $P(Y=1\mid X=c)$ via **MLE** (same as Phase 1).

**Target-dependent?** Yes. **Variance:** inherits Phase 1 — **high variance under low support**. **Leakage risk:** **high** if $\hat{p}_c$ is computed on data that includes the row’s own $Y$ (or test labels). **Cardinality:** one column per original feature; attractive for trees/linear models that want a scalar signal.

### 2.4 Smoothed target encoding

Replace $X=c$ with

$$
X^{(\mathrm{STE})} = \tilde{p}_c = \frac{k_c + \alpha}{n_c + \alpha + \beta},
$$

the Beta–Binomial **posterior mean** (Phase 1): same estimand $P(Y=1\mid X=c)$, **regularised** toward prior mean $\alpha/(\alpha+\beta)$.

**Target-dependent?** Yes. **Leakage risk:** **reduced** relative to extreme $\hat{p}_c$ on tiny $n_c$, but **still present** if computed with improper data scope (full test set, same fold). **Cardinality:** one column per feature.

---

## 3. Comparison table (article figure)

| Encoding | Formula | Estimates | Target-dependent? | Leakage risk | Cardinality handling |
|----------|---------|-----------|-------------------|--------------|----------------------|
| One-hot | $\mathbb{1}[X=c]$ | Nothing about $Y$ | No | None | Poor for high $C$ (wide sparse design) |
| Frequency | $n_c / N$ | $P(X=c)$ (empirical) | No | None | Good (single column) |
| Target (naïve) | $k_c / n_c$ | $P(Y=1\mid X=c)$ MLE | Yes | **High** if fit includes test/holdout or in-fold $Y$; worse when $n_c$ is small | Good |
| Smoothed target | $(k_c+\alpha)/(n_c+\alpha+\beta)$ | $P(Y=1\mid X=c)$ posterior mean | Yes | **Reduced** vs extreme MLE; still leaks if fit on wrong split | Good |

**Caption idea (draft):** *Each encoding answers a different statistical question. Only target-based encodings estimate $P(Y\mid X=c)$; they inherit estimator variance (Phase 1) and require correct training-only fitting to avoid leakage.*

---

## 4. Native categorical handling vs explicit encoding (2.2)

### What “native categoricals” means (tree boosting)

In **XGBoost / LightGBM / CatBoost** with categorical support, the learner can split on **sets of levels**, e.g. find $S \subseteq \{\text{levels}\}$ that minimises loss for a rule $X \in S$ vs $X \notin S$. **No fixed numeric encoding is required** for the algorithm to consume the feature — the tree searches over partitions of the category space.

### When encoding is **required**

- **Linear / logistic regression, NNs, SVMs, k-NN:** inputs must be numeric or fixed-size vectors → need one-hot, embeddings, target encoding, etc.
- **Pipelines that only accept `float` matrices:** same.

### When encoding is **optional but can add value**

Even with native trees, **target encoding** (properly cross-fitted) injects a **scalar** that approximates $P(Y\mid X=c)$. That can be **complementary** to raw category splits: the tree still learns interactions and threshold structure; the encoded column carries a **global** signal per level. Whether it helps is **empirical** (validation), not automatic.

### Model × encoding — decision matrix (sketch)

Rows: model family. Columns: role of encoding. Entries are **recommendations**, not laws — always validate.

| Model / setting | One-hot | Frequency | Target / smoothed target |
|-----------------|---------|-----------|---------------------------|
| Linear / logistic | Common default; watch $C$ | Possible single column | Strong signal; **must** be train-only / CV |
| Neural net | Embeddings often better than huge one-hot | Possible | Same leakage rules as linear |
| XGBoost / LGBM **native** cat | Optional | Optional | Optional complement; **CV** if used |
| Distance-based (k-NN) | Expand or embed | Scale carefully | Risky without proper CV |

**One paragraph answer (Exercise 2-B):** *If the booster handles categoricals natively, encoding is not required for correctness. You might still add target or smoothed target features when a scalar summary of $P(Y\mid X=c)$ helps the model converge faster or captures signal that greedy splits discover later; that is an empirical choice, and any target-based column must be fit without peeking at validation/test labels (cross-fitting), or you reintroduce the leakage problem below.*

---

## 5. Target leakage as consequence of improper encoding (2.3)

**Definition (one paragraph).** **Leakage** here means: the value fed to the model for a row **depends on that row’s label** (or on future data) in a way that **will not exist at deployment**. The model looks better than it is because it has seen the answer key through the feature pipeline.

**Mechanism (3 steps).**

1. Compute $\hat{p}_c = k_c/n_c$ (or smoothed variant) using **all** data — e.g. full train+test, or the same CV fold’s labels when encoding that fold.
2. For a row $i$ in category $c$, $k_c$ **includes** $Y_i$. A positive $Y_i$ **pushes** $\hat{p}_c$ upward for that row’s encoding → the feature **encodes its own label**.
3. At inference, you only have past data: encodings must come from **training statistics only**. The train-time pipeline used richer information → **optimistic bias** (gap between train and validation metrics).

**Link to Phase 1 (low support).** When $n_c$ is **small**, one row moves $\hat{p}_c$ a lot → **stronger per-row influence** → leakage and overfitting **worse** for rare categories.

**Fixes.**

- **$K$-fold / out-of-fold target encoding:** for each row, $\hat{p}_c$ from **other** folds only (sklearn `TargetEncoder` with CV).
- **Leave-one-out (LOO)** variants: exclude the current row from the count when encoding that row.

**Detection (one practical check).** After introducing target encoding, if **training AUC-PR** (or accuracy) jumps while **validation AUC-PR** barely moves or drops, suspect **leakage** or severe overfitting of the encoding — compare to a properly cross-fitted pipeline.

**Scope.** This is **not** a full taxonomy of leakage (Kaufman et al.); it is the slice that follows from **estimating $P(Y\mid X=c)$ on the wrong sample**.

---

## 6. References

- Micci-Barreca (2001) — target encoding with smoothing motivation.
- Kaufman et al. (2012) — leakage in data mining.
- scikit-learn `TargetEncoder` (cross-fitting); `category_encoders` smoothing parameters.
- XGBoost / LightGBM docs — categorical features and split finding.

---

## 7. Paper exercises (pencil and paper)

**Exercise 2-A — The encoding table**  
Reproduce the comparison table **from memory** on paper: formula, estimand, target-dependent?, leakage risk, cardinality. Compare to §3 above.

**Exercise 2-B — The XGBoost question**  
Write **one paragraph**: if boosting handles categoricals natively, **when and why** would you still add encodings? Under what conditions does target encoding add value? *(A draft answer is in §4.)*
