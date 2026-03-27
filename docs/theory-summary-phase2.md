# Theory summary (Phase 2) — article-ready blocks

Concise blocks for **Section 5 (encoding landscape)** and **Section 8 (leakage, brief)** in `article/features-that-lie.md`. Phase 1 remains the depth layer; Phase 2 is the **bridge to practice**.

---

## Block C1 — What does this encoding estimate?

- **One-hot** $\mathbb{1}[X{=}c]$: encodes **membership** in level $c$; **no** $Y$; no leakage from $Y$.
- **Frequency** $n_c/N$: estimates **prevalence** $\hat P(X{=}c)$; **no** $Y$.
- **Target (naïve)** $k_c/n_c$: MLE of **$P(Y{=}1\mid X{=}c)$**; inherits **variance under low $n_c$** (Phase 1); **leakage-prone** if fit on wrong data.
- **Smoothed target** $(k_c{+}\alpha)/(n_c{+}\alpha{+}\beta)$: **posterior mean** for the same estimand; **shrinkage** reduces extreme estimates; **leakage rules unchanged** — still must be fit **without** test labels and **without** in-fold self-use.

**Figure:** use the markdown table in `notes/phase2-encoding.md` §3 as the source for the article table/figure.

---

## Block C2 — Native categoricals vs encoding

Tree boosters with **native categorical** support split on **subsets of levels** ($X\in S$); **no numeric encoding is required** for the feature to be used. **Linear models, NNs, distance methods** require numeric or embedded inputs. **Optional** target-type encodings can still **add** a scalar summary of $P(Y\mid X{=}c)$ **alongside** raw categories; whether that helps is **empirical**, and target-derived columns demand **cross-fitted** estimation.

---

## Block C3 — Leakage in one breath

**Leakage:** the feature value for a row **depends on its own label** (or on data you will not have at prediction time). **Naïve target encoding** on pooled or test-inclusive data lets $Y_i$ **move** $\hat p_c$ for row $i$’s category → **inflated train metrics**. **Small $n_c$** (Phase 1) **amplifies** each row’s influence. **Fix:** out-of-fold / LOO computation of encodings. **Smoke test:** large **train vs validation AUC-PR** gap after adding target encoding → investigate **fit scope** first.

---

## Suggested section mapping

| Article section (roadmap) | Blocks |
|---------------------------|--------|
| §5 Encoding landscape | C1, C2 (table + short native discussion) |
| §8 Target leakage (brief) | C3 only — keep subsection length |
