# Phase 2 — Rough article drafts (notes)

Draft fragments for **Section 5** (encoding landscape) and **Section 8** (leakage, short). Lighter math than Phase 1. Merge into `article/features-that-lie.md` in Phase 4.

**Sources:** `notes/phase2-encoding.md`, `docs/theory-summary-phase2.md`.

---

## Section 5 — The encoding landscape (draft)

**Opening:** Every encoding function answers an implicit statistical question. Confusing those questions — treating a target-encoded value as “just a category ID with noise” — is how low-support categories and leakage bite in production.

**Body — four strategies (one short paragraph each):**  
One-hot: indicators; no $Y$; no leakage; dimension explodes with cardinality.  
Frequency: $n_c/N$ targets prevalence; useful when **rarity** correlates with fraud; no $Y$.  
Naïve target: $k_c/n_c$ is the MLE of $P(Y{=}1\mid X{=}c)$ from Phase 1 — variance and Wald pathologies carry over.  
Smoothed target: same estimand, Bayesian shrinkage toward the baseline; extremes (e.g. 5/5) are pulled toward sanity, but the column is still **built from $Y$** and needs correct fitting scope.

**Native trees:** XGBoost-style models can split on **sets of levels** without a hand-made numeric map. Encoding is then **optional**; target-type features are an **add-on** when a scalar $P(Y\mid c)$ summary helps — validate on held-out data.

**Figure:** Insert the **comparison table** (Encoding | Formula | Estimates | …) from the notes; caption: *What each encoding estimates, whether it uses $Y$, and leakage/cardinality trade-offs.*

**Close:** The right encoding is the one whose **estimand** matches the model’s needs and whose **variance and fit protocol** match the sample size and deployment story.

---

## Section 8 — Target leakage — when the estimator sees the answer (draft)

**Length target:** subsection weight — **one page** in final article, not a survey of all leakage types.

**Paragraph 1 — Definition:** Leakage means a feature’s value for a transaction **embeds information about that transaction’s label** (or about data you will not have when scoring), so offline metrics **lie**.

**Paragraph 2 — Mechanism:** Naïve target encoding pools counts $k_c, n_c$ **including** the current row. The encoded value **correlates with $Y_i$** by construction. Training looks excellent; validation and production do not repeat the cheat.

**Paragraph 3 — Low support:** When $n_c$ is tiny, one fraud flips $\hat{p}_c$ dramatically — the same **low-support** pathology as Phase 1, now doubled as **leakage amplitude**.

**Paragraph 4 — Fix and check:** Fit encodings **out-of-fold** or **leave-one-out** on training only. If train AUC-PR **spikes** vs validation after adding target encoding, **first** check whether $\hat{p}_c$ was computed with **forbidden** information.

**Closing line:** Leakage here is not a separate moral chapter; it is what happens when an **estimator of $P(Y\mid X=c)$** is evaluated on the **wrong sample**.

---

## Figure draft — Table only (no new asset file)

**Title (suggestion):** *Encoding strategies and what they estimate*

**Table:** Copy from `notes/phase2-encoding.md` §3 (full markdown table).

**Notes for layout:** In final article, consider transposing or splitting if wide; keep “Estimates” and “Leakage risk” columns — they carry the thesis.
