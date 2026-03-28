# Thesis, scope, and reader persona

## Thesis

Category-level statistics used in supervised fraud modelling—especially **target encodings** and **naïve** fraud rates \(\hat{p}_c = k_c/n_c\)—are **statistical estimators**, not fixed population truths. Their reliability is governed by **support** \(n_c\), the **scope** of the sample used to fit them (train vs test vs fold), and the **estimand** each encoding targets. Ignoring variance, smoothing, or leakage leads to two recurring illusions: (1) treating a **high observed** rate on a **small** \(n_c\) as decisive evidence of a precise population rate, and (2) treating **high Pearson correlation** between two encoded features as sufficient reason to drop one without **target-aware** or **model-based** evidence.

## Scope

This repository supports a **single narrative article** (`article/features-that-lie.md`) and reproducible **synthetic** experiments (`scripts/`, `src/`). It does **not** claim calibration on proprietary production data. It does **not** survey every encoding (e.g. full embedding pipelines). It **does** connect:

- Binomial MLE, Agresti–Coull intervals, Beta–Binomial smoothing  
- A compact encoding table (one-hot, frequency, naïve/smoothed target)  
- Correlation vs conditional redundancy (toy + pipeline)  
- Target leakage via wrong fit scope  

## Reader persona

- **Practitioners** building fraud scores who need language to push back on “perfect” rare categories and on correlation-only feature pruning.  
- **Interview / study** readers who want a coherent statistical story tied to runnable code.  
- **Reviewers** of the article who expect explicit links from claims to figures and to `docs/experiments-summary.md`.

## Relation to the portfolio site

The **canonical** article is Markdown in this repo. HTML for public reading is produced in a **separate** portfolio repository (Markdown → HTML). This repo remains the **source of truth** for text, BibTeX, and experiment code.
