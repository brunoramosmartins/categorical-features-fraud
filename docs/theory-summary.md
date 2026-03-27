# Theory summary (Phase 1) — article-ready blocks

Short, clean statements you can lift into `article/features-that-lie.md` after editing for voice and cross-references. Notation aligns with `article/notation.md` where symbols overlap.

---

## Block A1 — Category rate as an estimator

For level $c$ of feature $X$, with $k_c$ frauds in $n_c$ rows, $\hat{p}_c = k_c/n_c$ is the MLE of $p_c = P(Y{=}1\mid X{=}c)$ under a binomial model. Its variance is $\mathrm{Var}(\hat{p}_c) = p_c(1-p_c)/n_c$. Small $n_c$ implies **large variance** (**high-variance estimator under low support**). When $\hat{p}_c\in\{0,1\}$, plug-in variance is zero — a **Wald** artefact, not proof of certainty.

---

## Block A2 — Uruguay vs high-$n$ levels

For $k{=}n{=}5$, $\hat{p}{=}1$, Agresti–Coull uses $\tilde{p}=(k+2)/(n+4)=7/9$ and an adjusted sample size $n{+}4$ to build a 95% interval that remains **wide** (true rate far below 1 is still plausible). **Contrast:** for $n_c$ in the tens of thousands and $\hat{p}_c$ near the base rate, the same formula yields a **narrow** interval. **Takeaway:** “100% with five rows” is **more plausibly** a small-$n$ artefact than a reliable population signal.

---

## Block A3 — Smoothing as Beta–Binomial posterior mean

Prior $p_c\sim\mathrm{Beta}(\alpha,\beta)$ and $k_c\mid p_c\sim\mathrm{Binomial}(n_c,p_c)$ give posterior $p_c\mid\text{data}\sim\mathrm{Beta}(\alpha{+}k_c,\beta{+}n_c{-}k_c)$ with mean $(\alpha{+}k_c)/(\alpha{+}\beta{+}n_c)$. This equals a **convex combination** of the MLE $\hat{p}_c$ and the prior mean $\alpha/(\alpha{+}\beta)$ with weight $w_c=n_c/(n_c+\alpha+\beta)$: more shrinkage when $n_c$ is small, negligible when $n_c$ is large. **Interpretation:** “smoothing” is not an ad hoc trick; it is **Bayesian updating** with a conjugate prior whose strength sets how aggressively rare levels are pulled toward the baseline.

---

## Block B1 — Correlation $\neq$ redundancy

Two target-encoded columns can have Pearson correlation $>0.7$ while **conditional** probabilities $P(Y\mid X_1{=}a)$ and $P(Y\mid X_2{=}b)$ differ sharply for some $(a,b)$. A toy dataset with eight rows (three levels per feature) exhibits $\mathrm{Corr}(z_1,z_2)\approx0.72$ yet logistic regression on **both** encodings achieves higher accuracy than on **either** alone. **Conclusion:** **do not** drop features from pairwise correlation alone when encodings encode **target-related** information.

---

## Block B2 — From correlation matrix to decision

1. Flag high pairwise correlations among encodings.  
2. For each pair, score each feature against $Y$ (e.g. mutual information, permutation importance).  
3. If both carry signal, **keep both** and record the evidence.  
4. If one is inert, consider dropping it.  
5. Always **document** the metric and threshold.

This closes the gap: “we computed correlations” → “here is what we do about them, with respect to $Y$.”

---

## Suggested section mapping (article outline)

| Article section (roadmap) | Blocks |
|---------------------------|--------|
| §2 Estimator framing | A1 |
| §3 Low-support / Uruguay | A1, A2 |
| §4 Bayesian smoothing | A3 |
| §6 Correlation not redundancy | B1 |
| §7 When to drop / keep | B2 |

**Phase 2 (encoding & leakage):** see `docs/theory-summary-phase2.md` for §5 and §8 blocks.
