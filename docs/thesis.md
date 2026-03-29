# Thesis, scope, and reader persona

## Thesis

In **exploratory analysis**, categorical features require attention to **level counts** ($n_{c}$) and **target proportions** per level: a **high observed rate** on **few** rows is usually a **high-variance estimate**, not a reliable population fact. **Highly correlated predictors** (numeric, encoded, or mixed) should trigger **target-aware** and **model-based** checks — mutual information, permutation importance, validation ablation, and **VIF/regularisation** in linear models — not automatic feature drops on correlation alone.

Category-level statistics used in supervised modelling — especially **target encodings** and naïve rates $\hat{p}_{c} = k_{c}/n_{c}$ — are **statistical estimators**. Their reliability depends on **support** $n_{c}$, **fit scope** (train vs test vs fold), and the **estimand** each encoding targets. Ignoring variance, smoothing, or leakage produces **overconfident features** and **misleading training metrics**.

## Scope

Single narrative article (`article/features-that-lie.md`) plus reproducible **synthetic** experiments. Does not claim calibration on proprietary data; does not survey full embedding pipelines. Does connect binomial MLE, Agresti–Coull intervals, Beta–Binomial smoothing, encoding table, correlation vs **model** redundancy, and leakage.

## Reader persona

Practitioners who **profile categoricals in EDA** and need a validation checklist; ML engineers who see **high pairwise correlation** and must decide what to drop; reviewers who want explicit links from claims to figures and `docs/experiments-summary.md`.

## Relation to the portfolio site

Canonical article is Markdown in this repo; HTML for public reading is built in a **separate** portfolio repository.
