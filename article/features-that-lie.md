# When a Feature Looks Too Good to Be True

### Statistical Foundations of Categorical Feature Engineering for Fraud Detection — from Encoding to Inference

---

## Abstract

In fraud detection, a category with a 100% observed fraud rate on a handful of transactions is often treated as a near-perfect signal. A correlation matrix between engineered features is often computed—and then left without a decision rule. Both practices confuse **sample** quantities with **population** truths. This article treats category-level target rates and many encodings as **statistical estimators**: they have variances, failure modes under low support, and estimation protocols that determine whether they leak information from the label. We review maximum-likelihood estimation for binomial rates, Agresti–Coull intervals for small $n$, and Beta–Binomial (Bayesian) smoothing as the conjugate posterior mean. We unify common encodings by asking what population quantity each one estimates, explain why Pearson correlation between target-encoded features does not justify dropping one feature without target-aware evidence, and connect target leakage to computing those estimators on the wrong sample. Four reproducible experiments on a synthetic fraud dataset illustrate wide intervals for rare categories, smoothing and generalisation, a correlation-versus-redundancy trap, and train–test gaps under leaky target encoding. The closing section offers a compact decision lens: **every encoded value is an estimate—respect the sample size and the fit scope.**

**Keywords:** categorical features, target encoding, binomial estimation, Bayesian smoothing, leakage, fraud detection.

---

## 1. Introduction

During a technical interview for a fraud role, two moments stuck with me—not as failures, but as places where the analysis stopped short of the depth the problem deserved.

**First:** a country feature showed a 100% fraud rate for Uruguay. The instinct was to trust it; the interviewer’s guidance was to remove it. **Second:** a correlation matrix between features was produced; high correlations were noted, but there was no framework for what to do next—which feature to drop, on what principle, and with what evidence.

Those two moments look different on the surface. Underneath, they share one mistake: treating a **sample statistic** as if it were a **population parameter** with negligible uncertainty. A rate of 100% on five rows does not have the same evidentiary weight as a rate of 0.4% on forty thousand. A large Pearson correlation between two **target-derived** columns does not imply that one column is **redundant for predicting** the label.

This article’s thesis is statistical, not categorical:

> *A category with 100% target rate and five observations is **more likely** evidence of **insufficient data** than of a robust predictive signal. A high correlation between two encoded features is **more likely** evidence of **shared variation** than of redundancy in how they inform the target. Both illusions stem from ignoring that every encoded value used in supervised learning is an **estimator**—with variance, bias conditions, and a correct sample on which it must be fit.*

The central axis is simple: **every category-level statistic used in feature engineering is an estimator.** The sample size $n_c$ at level $c$ controls how much trust $\hat{p}_c = k_c/n_c$ deserves. Smoothing is not mere regularisation—it is the posterior mean under a Beta–Binomial model when that is the chosen prior [3,4]. Leakage is not a separate moral chapter—it is what happens when an estimator of $P(Y\mid X=c)$ is computed using information that will not exist at deployment [6].

**Roadmap.** Section 2 frames the MLE and its variance. Section 3 deconstructs the “Uruguay” pattern with Agresti–Coull intervals [1,2]. Section 4 gives Beta–Binomial smoothing as a weighted average of data and prior. Section 5 catalogues encodings by **estimand**. Section 6 gives a formal correlation-versus-redundancy counterexample. Section 7 proposes a target-aware decision heuristic after computing correlations. Section 8 briefly treats target leakage. Section 9 summarises four experiments (Figures 1–4). Section 10 synthesises a decision lens. Section 11 concludes.

**Who should read this.** Fraud and risk modellers who ship categorical features into production; data scientists preparing for interviews where “100% on five rows” and “correlation matrices” appear; and readers who want a **single** statistical narrative with **reproducible** code rather than a library survey.

---

## 2. A category-level statistic is an estimator

Let $X$ denote a categorical feature and $c$ a fixed level. Among the $n_c$ training rows with $X=c$, let $k_c$ be the number with $Y=1$ (fraud). The **sample proportion**

$$
\hat{p}_c = \frac{k_c}{n_c}
$$

is the **maximum likelihood estimator** (MLE) of $p_c = P(Y=1\mid X=c)$ under a binomial model: conditional on $X=c$, each $Y$ is Bernoulli$(p_c)$ [7].

The sampling variance (conditioning on $n_c$) is

$$
\mathrm{Var}(\hat{p}_c) = \frac{p_c(1-p_c)}{n_c}.
$$

**Low support** means $n_c$ is small, so the variance is large: $\hat{p}_c$ is a **high-variance estimator under low support**. The **observed** value can be extreme even when $p_c$ is moderate.

A separate pathology appears for **plug-in** variance estimates $\hat{p}_c(1-\hat{p}_c)/n_c$: when $\hat{p}_c\in\{0,1\}$, this expression is **zero**, suggesting—falsely—that there is no uncertainty. Interval methods aimed at binomial proportions (Wilson, Agresti–Coull, Clopper–Pearson) remain wide in that regime [1,2].

**Consequence.** Target encoding that maps level $c$ to $\hat{p}_c$ does not produce a “true fraud propensity” stamped on the category; it produces an **estimate** whose reliability is governed by $n_c$.

**Asymptotics and modelling.** For large $n_c$, the MLE is approximately normal with the variance above, and standard Wald intervals become usable when $\hat{p}_c$ is not on the boundary. In fraud settings, **most** levels may have moderate $n_c$ while a **long tail** of rare levels drives the policy question—precisely where normal approximations and plug-in variance fail simultaneously. The engineering takeaway is not “never use $\hat{p}_c$,” but **never confuse $\hat{p}_c$ with $p_c$** when $n_c$ is small.

---

## 3. The low-support problem: a numerical deconstruction

Take the narrative anchor: **Uruguay**, with about **five** transactions, all fraudulent: $k_c=n_c=5$, hence $\hat{p}_c=1$.

The Agresti–Coull adjusted proportion uses pseudo-counts:

$$
\tilde{p} = \frac{k+2}{n+4}, \qquad \tilde{n} = n+4.
$$

For $(k,n)=(5,5)$, $\tilde{p}=7/9\approx 0.78$. An approximate 95% interval uses $\tilde{p} \pm z_{0.975}\sqrt{\tilde{p}(1-\tilde{p})/\tilde{n}}$, yielding a wide band—often extending down to roughly **0.5** and up to **1** after clipping [1]. The headline “100% fraud” is **compatible** with a true $p_c$ far below 1.

Contrast a large country such as **Brazil**: if $n_c$ is on the order of $10^4$ and $\hat{p}_c\approx 0.004$, the same interval machinery produces a **narrow** band (width on the order of $10^{-3}$). The estimator is informative because **$n_c$ is informative**.

**Punchline (softened).** The observed perfection for Uruguay is **more plausibly** an artefact of **small $n$** than evidence of a **stable** extreme population rate—exactly the statistical phrasing you want in a room where “100%” sounds decisive.

| Anchor | $n_c$ (illustrative) | $k_c$ | $\hat{p}_c$ | 95% AC interval (order of magnitude) |
|--------|----------------------|-------|-------------|----------------------------------------|
| Uruguay | $\approx 5$ | $\approx 5$ | $1.0$ | Wide (e.g. lower endpoint $\approx 0.5$) |
| Brazil | $\approx 10^4$ | $\approx 0.004\,n_c$ | $\approx 0.004$ | Narrow ($\sim 10^{-3}$ width) |

The numbers in the Brazil row are **illustrative** of a high-$n$ level near the global base rate; your dataset will differ. The **structure** is the point: the same statistical question (“what is $p_c$?”) has radically different **precision** across levels.

---

## 4. Bayesian smoothing: the principled fix

Place a Beta prior on $p_c$:

$$
p_c \sim \mathrm{Beta}(\alpha,\beta),
$$

and likelihood $k_c \mid p_c \sim \mathrm{Binomial}(n_c, p_c)$. The posterior is conjugate:

$$
p_c \mid k_c, n_c \sim \mathrm{Beta}(\alpha + k_c,\; \beta + n_c - k_c),
$$

with **posterior mean**

$$
\tilde{p}_c^{\mathrm{Bayes}} = \frac{\alpha + k_c}{\alpha + \beta + n_c}.
$$

Write $m=\alpha+\beta$ and prior mean $\mu_0=\alpha/(\alpha+\beta)$. Then

$$
\tilde{p}_c^{\mathrm{Bayes}} = w_c \hat{p}_c + (1-w_c)\mu_0,
\qquad
w_c = \frac{n_c}{n_c + \alpha + \beta}.
$$

When $n_c$ is small, $w_c$ is small: the estimate **shrinks** toward $\mu_0$ (e.g. the global fraud rate). When $n_c$ is large, $w_c\to 1$: the posterior mean **tracks** the MLE [3,4].

**Interpretation.** Smoothing parameters in libraries such as `category_encoders` can be read as **prior strength** relative to $n_c$—not as an arbitrary knob [5].

**Choosing $(\alpha,\beta)$.** A common default is to set the prior mean $\mu_0=\alpha/(\alpha+\beta)$ to the **global** training fraud rate $\bar{p}$, and to select total prior mass $m=\alpha+\beta$ by cross-validation or domain guidance: larger $m$ pulls rare levels harder toward $\bar{p}$. This mirrors empirical-Bayes practice: the prior is not “belief” in a metaphysical sense but a **regulariser** whose strength is tuned against held-out performance—while remaining interpretable as a Beta posterior mean [3,4].

**Connection to production.** At scoring time, new rows with a rare level $c$ still use counts $(k_c,n_c)$ from **training** (or from a rolling training window). Smoothing stabilises the encoded value sent to the model when $n_c$ is tiny; it does **not** remove the need to monitor support for $c$ over time.

---

## 5. The encoding landscape: what does each map estimate?

| Encoding | Formula (level $c$) | Estimates | Uses $Y$? | Leakage risk | High cardinality |
|----------|---------------------|-----------|-----------|--------------|------------------|
| One-hot | $\mathbb{1}[X=c]$ | Membership in $c$ | No | None | Poor (wide sparse design) |
| Frequency | $n_c/N$ | $\hat{P}(X=c)$ | No | None | Good (one column) |
| Target (naïve) | $k_c/n_c$ | $P(Y{=}1\mid X{=}c)$ MLE | Yes | **High** if misfit | Good |
| Smoothed target | $(k_c{+}\alpha)/(n_c{+}\alpha{+}\beta)$ | Same estimand, posterior mean | Yes | Reduced vs extremes; still misfit if wrong scope | Good |

Tree boosters with **native categorical** support search splits such as $X\in S$ for subsets $S$ of levels; **no hand-built numeric encoding is required** for the learner to use the feature. Target-type encodings can still add a **scalar summary** of $P(Y\mid X{=}c)$ alongside raw categories; whether that helps is **empirical**, and target-derived columns must be fit with **cross-fitting** when labels must not leak across folds [7].

**Frequency encoding** estimates prevalence $P(X{=}c)$, not $P(Y\mid X{=}c)$. It carries **no** label information and therefore **no** target leakage from the encoding map itself. It can still help tree and linear models when **rarity** of a level is predictive of $Y$ even if the **naïve** fraud rate at that level is unstable.

**Embeddings** (learned dense vectors for levels) are outside this article’s scope: they introduce a different estimand and optimisation loop. The estimator lens still applies if embeddings are trained with label information—**fit scope** and **leakage** remain central.

---

## 6. Correlation is not redundancy

The second narrative anchor: a **correlation matrix** was computed; high correlation was observed; the analysis **stopped** before a target-aware decision.

**Fact.** Pearson correlation between two **target-encoded** columns measures **linear association across rows** between the two level-wise estimates assigned to each row. It does **not** imply that the two features are **conditionally redundant** for $Y$: $P(Y\mid X_1)$ and $P(Y\mid X_2)$ can differ sharply for specific pairs of levels even when the encoded columns correlate.

**Worked toy (eight rows).** Levels $X_1\in\{A,B,C\}$, $X_2\in\{M,N,P\}$, binary $Y$:

| Row | $X_1$ | $X_2$ | $Y$ |
|-----|-------|-------|-----|
| 1 | C | M | 0 |
| 2 | A | P | 1 |
| 3 | C | M | 0 |
| 4 | B | P | 1 |
| 5 | B | M | 0 |
| 6 | B | M | 1 |
| 7 | B | P | 0 |
| 8 | C | N | 0 |

Naïve training target encodings yield $\mathrm{Corr}(z_1,z_2)\approx 0.72$, yet $P(Y{=}1\mid X_1{=}A)=1$ while $P(Y{=}1\mid X_2{=}M)=1/4$, and a logistic model on **both** encodings can outperform either alone on this table. **Marginal correlation is not conditional redundancy.**

**Pipeline experiments.** On a large synthetic fraud dataset (see §9), Pearson correlation between **row-wise** target encodings of `country` and `merchant_category` may fall **below** $0.7$ because many levels **pool** rows and dilute row-level linear association. The **principle** remains: inspect **conditional** behaviour and **model scores** tied to $Y$, not $|r|$ alone. The toy table is the **proof pattern**; the codebase is the **stress test**.

---

## 7. When to drop, when to keep

**Unsafe (for target-derived encodings).** Dropping one feature because $|r|>0.9$ with another **without** checking relationship to $Y$.

**Safer screens.** Mutual information with $Y$, permutation importance, or model explanations—these ask whether the feature **moves** the prediction of $Y$, not merely whether it tracks another feature [7].

**Five-step heuristic** (after computing correlations):

1. Flag high $|r|$ among encoded columns.
2. For each flagged pair, score **each** feature against $Y$ (e.g. mutual information or permutation importance).
3. If **both** carry signal, **keep both** unless a simpler model is required—document why.
4. If one is inert, consider dropping the inert one—document the metric.
5. **Write the decision** with numbers, not “we removed collinear features.”

This is what was missing when the case stopped at “we saw high correlation.”

---

## 8. Target leakage: when the estimator sees the answer

**Leakage** (narrow sense used here): a feature’s value for a row **depends on that row’s label** (or on future data) in a way that will not exist at scoring time.

**Mechanism (naïve target encoding).** Compute $\hat{p}_c=k_c/n_c$ using **all** rows, including the row being encoded. The count $k_c$ **includes** the current row’s $Y_i$, so the encoding **encodes the answer key** for that row. Training metrics inflate; the proper comparison is **out-of-fold** or leave-one-out statistics fit **without** the row’s label, or strictly **training-only** scopes [6].

**Low support amplifies severity.** When $n_c$ is tiny, one label moves $\hat{p}_c$ sharply—so the self-influence is large.

**Detection smoke test.** If training AUC-PR jumps while validation AUC-PR barely moves after introducing target encoding, **first** check whether encodings were fit with forbidden label information.

---

## 9. Experiments

**Reproducibility.** From the repository root, with Python 3.10+:

```text
python -m venv .venv && source .venv/bin/activate   # or Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py
```

Figures are written to `figures/` at the DPI set in `config.yaml` (default 300). The synthetic generator is documented in `docs/dataset-design.md`; caveats and numeric ranges are summarised in `docs/experiments-summary.md`.

### Experiment A — The “perfect feature” illusion

**Setup.** Stratified train/test split; **all** Uruguay rows are kept in **training** so the low-support anchor has $n_c{=}5$, $k_c{=}5$ in-train. Per-country naïve target rates $\hat{p}_c$ and 95% Agresti–Coull intervals are computed on the training split. A second panel varies prior strength $m=\alpha+\beta$ for a Beta prior centred at the global training fraud rate.

**Figure 1** (`figures/exp_a_perfect_feature.png`). **Left:** $\hat{p}_c$ versus $n_c$ (log scale) with error bars; Uruguay annotated. **Right:** smoothed posterior mean for Uruguay versus $m$, with horizontal line at $\bar{p}$.

**Observation.** The interval for Uruguay remains wide despite $\hat{p}_c{=}1$; increasing $m$ pulls the smoothed estimate toward $\bar{p}$.

**Theory link.** §2–§4: MLE variance, Agresti–Coull, Beta–Binomial shrinkage.

### Experiment B — Smoothing and generalisation

**Setup.** XGBoost on numerical features plus **country** encoded three ways: naïve target, smoothed target (hyperparameters from `config.yaml`), and one-hot. Metrics: train/test AUC-PR and F1 at threshold $0.5$.

**Figure 2** (`figures/exp_b_smoothing_effect.png`). Test AUC-PR by encoding; grouped bars for train vs test AUC-PR.

**Observation.** Naïve target encoding often shows a **larger** train–test gap than smoothed encoding; exact ranking varies with seed and model hyperparameters.

**Theory link.** §4: variance reduction via shrinkage; §5: what each encoding estimates.

### Experiment C — Correlation versus redundancy

**Setup.** Naïve target encodings for `country` and `merchant_category` fit on **training only**, applied to train/test rows. Pearson $r$ on training rows; mutual information with $Y$; XGBoost with numerics + both encodings vs dropping one column.

**Figure 3** (`figures/exp_c_correlation_trap.png`). Scatter of encodings; bar chart of test AUC-PR for {both, country only, merchant only}; MI bar chart.

**Observation.** High $|r|$ does **not** license dropping a feature without target-aware evidence; in some draws, one encoding **approximates** the signal of the other—then the bar chart reflects **overlap**, not a failure of logic. The **decision rule** remains §7.

**Theory link.** §6–§7.

### Experiment D — Leakage via encoding scope

**Setup.** **Leaky:** naïve target encoding for `country` computed using **concatenated** train and test labels. **Proper:** out-of-fold encoding on training rows; test rows encoded from **training-only** statistics. Same XGBoost hyperparameters otherwise. Experiment script may use a **larger** effective test fraction to amplify the pooled-label effect (see `scripts/experiment_d_encoding_comparison.py`).

**Figure 4** (`figures/exp_d_encoding_comparison.png`). Train vs test AUC-PR for leaky vs proper pipelines.

**Observation.** Train AUC-PR can be **optimistic** under leakage; the gap pattern supports §8.

**Theory link.** §8; estimator fit scope [6].

---

## 10. A framework for encoding decisions

Think in three axes: **cardinality** (how many levels), **support** (typical $n_c$), and **model family** (linear/NN vs tree with native categoricals).

- **High cardinality, low support for many levels:** prefer **smoothed target** or **frequency** plus regularised models; avoid interpreting raw $\hat{p}_c$ on tiny $n_c$ as ground truth.
- **Linear models / NNs:** need numeric or embedded inputs; **one-hot** or **target-type** encodings are typical; target-type requires **cross-fitting**.
- **Gradient-boosted trees with native categories:** encoding optional; target encodings may still add a global scalar signal—validate.
- **After any correlation matrix on encodings:** run the **§7** heuristic before dropping features.

The table below is a **compass**, not a law: always validate on held-out data.

| Cardinality | Typical support | Model (examples) | First-line encoding | Fit scope for any target-based map |
|-------------|-----------------|------------------|---------------------|-----------------------------------|
| Low | High per level | Logistic regression | One-hot or target (smooth) | Train only; CV for target |
| High | Heavy tail of small $n_c$ | XGBoost (native cat) | Native + optional smoothed target | OOF / train-only for target |
| High | Heavy tail | Neural network | Embedding or frequency + numeric | Embeddings: avoid label leakage in training graph |
| Any | Any | Any | After high pairwise $|r|$ among encodings | Run §7 **before** dropping |

**Closing line.** Every encoded value is an estimate. Treat it with the **statistical respect** the sample size and fit scope demand.

---

## 11. Conclusion

We started from two interview moments: a **100%** rate on **five** rows, and a **correlation matrix** without a decision rule. Both are symptoms of the same oversight—confusing **estimates** with **truths**.

The article demonstrated: (i) the MLE $\hat{p}_c$ and its variance under low support; (ii) Agresti–Coull intervals that keep uncertainty visible when $\hat{p}_c$ hits 0 or 1; (iii) Beta–Binomial smoothing as a posterior mean with a transparent data-versus-prior weight; (iv) encodings catalogued by **estimand**; (v) a correlation trap where redundancy is **not** implied; (vi) a target-aware heuristic for feature decisions; (vii) leakage as wrong estimation scope; (viii) four reproducible figures tying claims to code.

**Conditional takeaways.** If $n_c$ is small, soften language and widen intervals before trusting extreme rates. If two target encodings correlate highly, **measure each against $Y$** before dropping one. If you use target encoding, **define the sample** on which $k_c$ and $n_c$ are computed as carefully as you define the train/test split.

**What comes next.** Apply the same estimator lens to **sequential** or **production** encodings (concept drift), to **hierarchical** priors across related categories, and to **monitoring** $n_c$ over time so rare levels do not silently become high-variance again.

**Limitations.** All quantitative claims tied to AUC-PR, F1, and correlation magnitudes are **conditional** on the synthetic generator in `src/data.py` and hyperparameters in `config.yaml`. The article does **not** assert that smoothed target encoding always beats one-hot on real fraud data, nor that Experiment C will always show $|r|>0.7$. The **logical** claims—variance under low support, interval behaviour at boundaries, wrong-scope leakage, and correlation $\not\Rightarrow$ conditional redundancy for $Y$—hold regardless of the synthetic draw.

---

## Appendix A: Repository file map

| Article locus | Code / docs |
|---------------|-------------|
| §3–§4, Fig. 1 | `scripts/experiment_a_perfect_feature.py`, `src/stats_utils.py` |
| §4–§5, Fig. 2 | `scripts/experiment_b_smoothing_effect.py`, `src/encoding.py`, `src/models.py` |
| §6–§7, Fig. 3 | `scripts/experiment_c_correlation_trap.py` |
| §8, Fig. 4 | `scripts/experiment_d_encoding_comparison.py` |
| Generator | `src/data.py`, `docs/dataset-design.md` |
| Numeric summary | `docs/experiments-summary.md` |
| Symbols | `article/notation.md` |
| BibTeX | `article/references.bib` |

Run order for all figures: `python scripts/run_all.py` (see `CONTRIBUTING.md` for branch conventions).

---

## References

[1] Agresti, A., & Coull, B. A. (1998). Approximate is better than “exact” for interval estimation of binomial proportions. *The American Statistician*, 52(2), 119–126.

[2] Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101–133.

[3] Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.

[4] Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

[5] Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes. *ACM SIGKDD Explorations*, 3(1), 27–32.

[6] Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining. *ACM TKDD*, 6(4), 1–21.

[7] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

*BibTeX keys:* `agresti1998approximate`, `brown2001interval`, `bishop2006pattern`, `murphy2012machine`, `micci2001preprocessing`, `kaufman2012leakage`, `hastie2009elements` — see `article/references.bib`.
