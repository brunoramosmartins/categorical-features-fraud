# Phase 1 — Rough article drafts (notes)

Draft fragments for Sections **2, 3, 4, 6, 7** (roadmap). Style: claim → derivation → consequence. Polish, shorten, and merge into `article/features-that-lie.md` in Phase 4.

**Source of truth for math:** `notes/phase1-theory.md` and `docs/theory-summary.md`.

---

## Section 2 — A category-level statistic is an estimator (draft)

**Claim:** Every number you attach to a category — especially after target encoding — is an **estimate** of a population quantity, not a fact.

**Setup:** Fix level $c$ of $X$. Among $n_c$ transactions with $X=c$, count $k_c$ frauds. Model $k_c \mid p_c \sim \mathrm{Binomial}(n_c, p_c)$ with $p_c = P(Y=1\mid X=c)$.

**Derivation:** The MLE is $\hat{p}_c = k_c/n_c$. Variance $\mathrm{Var}(\hat{p}_c)=p_c(1-p_c)/n_c$.

**Consequence:** Small $n_c$ → large variance → **high-variance estimator under low support**. If $\hat{p}_c=1$, plug-in variance is zero — interval methods must not be Wald-only.

---

## Section 3 — The low-support problem: Uruguay (draft)

**Claim:** A “100% fraud rate” on ~five rows does **not** license treating that category as a near-perfect fraud detector.

**Numerics:** $k=n=5$, $\hat{p}=1$. Agresti–Coull adjusted $\tilde{p}=7/9$; 95% interval still spans most of $[0.5,1]$ (see notes for exact margin).

**Contrast:** Brazil-scale $n_c$ gives interval width $\sim 10^{-3}$ for $\hat{p}\approx 0.004$.

**Consequence (wording):** The observed rate is **more plausibly** consistent with **insufficient data** than with a **stable** extreme population probability — the honest statistical phrasing matters for interviews and production.

---

## Section 4 — Bayesian smoothing (draft)

**Claim:** Smoothing rare categories toward the global mean is **Bayesian shrinkage** with a clear weight on data vs prior.

**Derivation:** Beta prior + binomial likelihood → Beta posterior; mean $(\alpha+k_c)/(\alpha+\beta+n_c) = w_c \hat{p}_c + (1-w_c)\mu_0$ with $w_c=n_c/(n_c+\alpha+\beta)$.

**Consequence:** Encoding libraries that add “pseudocounts” mirror this structure; the **tuning knob** is prior strength relative to $n_c$, not magic.

---

## Section 6 — Correlation is not redundancy (draft)

**Claim:** High Pearson correlation between two **target-encoded** columns does **not** mean one column is redundant for predicting $Y$.

**Evidence:** Eight-row toy table (see `notes/phase1-theory.md`): $\mathrm{Corr}(z_1,z_2)\approx 0.72$, yet $P(Y|X_1=A)\neq P(Y|X_2=M)$, and a two-feature logistic model beats either single-feature model on accuracy.

**Consequence:** Feature dropping by correlation thresholds is **unsafe** for target-derived encodings; you need **target-aware** evidence.

---

## Section 7 — When to drop, when to keep (draft)

**Claim:** The missing step after a correlation matrix is a **rule that talks about $Y$**.

**Heuristic (numbered):** (1) Flag high $|r|$. (2) MI or permutation importance vs $Y$ for each member of the pair. (3) Both substantial → keep both. (4) One inert → candidate to drop. (5) Write the memo with numbers.

**Consequence:** This is the completion of the interview gap: from **observation** to **documented decision**.
