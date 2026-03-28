# When a Feature Looks Too Good to Be True

### Statistical Foundations of Categorical Feature Engineering for Fraud Detection — from Encoding to Inference

*Second draft — structured technical review pass (clarity, external validity, operational checklists).*

---

## Abstract

In fraud and risk modelling, a category with a **high observed event rate** on few transactions is often treated as a strong signal—inviting **bias** and **overfitting** when that rate is mistaken for a precise population fact. Correlation matrices between engineered features are often produced without a rule for what to do next. Both habits confuse **sample** quantities with **population** truths. This article treats category-level rates and common encodings as **statistical estimators** with variance, low-support failure modes, and correct **fit scope** (including leakage). We outline MLE for binomial rates, Agresti–Coull intervals for small $n$, and Beta–Binomial smoothing; we unify encodings by what they estimate; and we explain why high Pearson correlation between target-encoded columns does not justify dropping a feature without target-aware evidence. Four reproducible experiments on a **synthetic** dataset illustrate intervals for rare levels, smoothing and generalisation, correlation versus redundancy for **model** performance, and train–test gaps under leaky encoding—**synthetic extremes are pedagogical, not guaranteed in production**. The article closes with explicit checklists for feature decisions and deployment.

**Keywords:** categorical features, target encoding, binomial estimation, Bayesian smoothing, leakage, fraud detection, overfitting.

---

## 1. Introduction

**Context.** In many technical case studies and production pipelines, categorical features drive both explanation (“this segment looks risky”) and model inputs. Practitioners routinely compute level-wise event rates and correlation matrices between encoded columns.

**Problem.** Two recurring gaps show up in review. First, a rare level displays a **very high observed rate** (sometimes 100%, often 70–95% on handfuls of rows). The team must decide whether to trust it, cap it, smooth it, or drop it—often without quantifying uncertainty. Second, two encoded features correlate highly; someone suggests dropping one for “collinearity,” without asking what each column contributes to predicting the label.

**Risk.** Treating $\hat{p}_c=k_c/n_c$ as if it were $p_c$ with negligible error leads to **overconfident features** and **overfitting** on noise in rare levels. Using pairwise correlation alone for target-derived columns can **remove real signal** from the model.

**Motivation.** The fix is not a new library call in isolation—it is a **statistical stance**: every encoded value is an **estimate** whose precision depends on $n_c$ and on **where** counts were computed (train vs test vs fold). This article makes that stance operational: formulas where they help, experiments where they convince, and **checklists** where they ship.

This article’s thesis is statistical, not categorical:

> *A category with a **high observed** target rate and **few** observations is **more likely** evidence of **high estimator variance** (and possibly insufficient data) than of a rate you should treat as tightly known. A high correlation between two encoded features is **more likely** evidence of **shared variation** than of safe redundancy for predicting $Y$. Both mistakes ignore that supervised encodings are **estimators**—with variance, bias risk under misspecification, and a correct sample on which they must be fit.*

The central axis: **every category-level statistic used in feature engineering is an estimator.** The sample size $n_c$ at level $c$ governs how much trust $\hat{p}_c = k_c/n_c$ deserves. Smoothing is the Beta–Binomial posterior mean when that model is the chosen prior [3,4]. Leakage is computing an estimator of $P(Y\mid X=c)$ with information unavailable at scoring time [6].

**Roadmap.** Section 2 frames the MLE and variance (after intuition). Section 3 deconstructs low support with Agresti–Coull intervals [1,2] and **non-extreme** examples. Section 4 gives Bayesian smoothing, library mapping, and pseudocode. Section 5 catalogues encodings and **when to use** them. Section 6: correlation versus redundancy, with emphasis on **models**. Section 7: a **checklist** after correlations. Section 8: leakage with a minimal example. Section 9: experiments and a **summary table** (paths to figures relative to repo root). Section 10: decision framework and flow. Section 11 concludes. **Appendix B** lists a **production-oriented checklist**.

**Who should read this.** Fraud and risk modellers shipping categorical features; ML engineers auditing encoding and leakage; readers who want one coherent estimator narrative plus runnable code.

**Synthetic data caveat.** The repository’s generator can exhibit **stylised** extremes (e.g. a tiny level with all positives) to make plots readable. **Production** more often shows **high but not perfect** rates under low $n$; the **same statistics** apply.

---

## 2. A category-level statistic is an estimator

**Intuition first.** If almost nothing has been observed at level $c$, the empirical rate $\hat{p}_c$ is a **noisy dial**: it can swing to 0.9 or 1.0 by chance even when the long-run rate is moderate. **Few rows → high variance** in $\hat{p}_c$. Feature engineering that plugs $\hat{p}_c$ into a model therefore injects **high-variance inputs** unless you smooth, regularise, or pool information.

Now the formal framing. Let $X$ denote a categorical feature and $c$ a fixed level. Among the $n_c$ training rows with $X=c$, let $k_c$ be the number with $Y=1$ (fraud). The **sample proportion**

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

**Consequence.** Target encoding that maps level $c$ to $\hat{p}_c$ does not produce a “true fraud propensity” stamped on the category; it produces an **estimate** whose reliability is governed by $n_c$. Feeding it unchecked into flexible models raises **overfitting** risk on rare levels.

**Asymptotics and modelling.** For large $n_c$, the MLE is approximately normal with the variance above. In fraud, a **long tail** of rare levels often drives policy—precisely where normal and plug-in shortcuts fail together. The engineering takeaway: **never confuse $\hat{p}_c$ with $p_c$** when $n_c$ is small.

---

## 3. The low-support problem: a numerical deconstruction

The running example uses a rare level **Uruguay** in **synthetic** data: about five in-sample rows, all positives, so $k_c=n_c=5$ and $\hat{p}_c=1$. **Real pipelines** more commonly see **high** rates (e.g. 0.75–0.95) on **small** $n_c$; the interval logic below applies unchanged—the boundary case $\hat{p}_c=1$ is the **clearest** illustration of plug-in variance failure, not the only case that matters.

The Agresti–Coull adjusted proportion uses pseudo-counts:

$$
\tilde{p} = \frac{k+2}{n+4}, \qquad \tilde{n} = n+4.
$$

For $(k,n)=(5,5)$, $\tilde{p}=7/9\approx 0.78$. An approximate 95% interval uses $\tilde{p} \pm z_{0.975}\sqrt{\tilde{p}(1-\tilde{p})/\tilde{n}}$, yielding a wide band—often extending down to roughly **0.5** and up to **1** after clipping [1]. A headline rate near **1** is **compatible** with a true $p_c$ far below 1.

For a **less extreme** illustration, take $(k,n)=(7,10)$: $\hat{p}_c=0.7$. The Agresti–Coull interval is still **wide** relative to large-$n$ settings; the point is that **moderately high** $\hat{p}_c$ on **tens** of rows still carries substantial uncertainty compared with levels with thousands of rows.

Contrast a large country such as **Brazil**: if $n_c$ is on the order of $10^4$ and $\hat{p}_c\approx 0.004$, the same interval machinery produces a **narrow** band (width on the order of $10^{-3}$). The estimator is informative because **$n_c$ is informative**.

**Punchline (softened).** Extreme **observed** rates at tiny $n_c$ are **more plausibly** driven by **sampling noise** than by a **tightly known** population rate—whether the observed value is 1.0 or 0.85.

| Anchor | $n_c$ (illustrative) | $k_c$ | $\hat{p}_c$ | 95% AC interval (order of magnitude) |
|--------|----------------------|-------|-------------|----------------------------------------|
| Rare level (synthetic) | $\approx 5$ | $\approx 5$ | $1.0$ | Wide (e.g. lower endpoint $\approx 0.5$) |
| Rare level (moderate) | $10$ | $7$ | $0.7$ | Wide (substantial width) |
| Brazil | $\approx 10^4$ | $\approx 0.004\,n_c$ | $\approx 0.004$ | Narrow ($\sim 10^{-3}$ width) |

The Brazil row is **illustrative**. The **structure** is the point: “what is $p_c$?” has different **precision** by level.

**Practical rule of thumb (not a law).** Before treating $\hat{p}_c$ as a feature “truth,” report **$n_c$** alongside it; for small $n_c$, prefer **intervals** or **smoothed** values. Many teams treat $n_c$ below a few dozen (or below a domain-specific minimum) as **low-support** for stable rate estimation—tune the threshold with **held-out model performance**, not superstition.

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

**Libraries (practice bridge).** In `category_encoders`, smoothing parameters for target-like encoders are usefully read as **prior strength** relative to $n_c$ [5]. In **scikit-learn** 1.2+, `TargetEncoder` performs **cross-fitted** target statistics to reduce leakage—conceptually aligned with the **fit scope** discipline below, even when the parametric story is not Beta–Binomial.

**Pseudocode (smoothed level map, train → apply).**

```text
global_mean ← mean(y_train)
for each level c observed in X_train:
    (k_c, n_c) ← counts of Y=1 and rows with X=c on training
    map[c] ← (alpha + k_c) / (alpha + beta + n_c)
for each row i in X_apply:
    x ← category of row i
    encoded[i] ← map[x] if x in map else global_mean
```

**Choosing $(\alpha,\beta)$.** A common default sets $\mu_0=\alpha/(\alpha+\beta)$ to the **global** training rate $\bar{p}$, and chooses $m=\alpha+\beta$ by cross-validation or domain guidance: larger $m$ pulls rare levels harder toward $\bar{p}$ [3,4].

**Connection to production.** At scoring time, rare level $c$ uses $(k_c,n_c)$ from **training** (or a rolling training window). Smoothing stabilises the encoded value; it does **not** remove the need to monitor $n_c$ over time.

---

## 5. The encoding landscape: what does each map estimate?

| Encoding | Formula (level $c$) | Estimates | Uses $Y$? | Leakage risk | High cardinality |
|----------|---------------------|-----------|-----------|--------------|------------------|
| One-hot | $\mathbb{1}[X=c]$ | Membership in $c$ | No | None | Poor (wide sparse design) |
| Frequency | $n_c/N$ | $\hat{P}(X=c)$ | No | None | Good (one column) |
| Target (naïve) | $k_c/n_c$ | $P(Y{=}1\mid X{=}c)$ MLE | Yes | **High** if misfit | Good |
| Smoothed target | $(k_c{+}\alpha)/(n_c{+}\alpha{+}\beta)$ | Same estimand, posterior mean | Yes | Reduced vs extremes; still misfit if wrong scope | Good |

**When to use (compact).**

- **One-hot:** low cardinality, linear models, interpretable level effects; avoid on very high $C$ without regularisation.
- **Frequency:** high cardinality, signal when **rarity** of $X$ matters; no label leakage from the map.
- **Naïve target:** rarely appropriate for final training without **OOF / CV**; useful as a teaching contrast.
- **Smoothed target:** high cardinality with label signal; tune smoothing; always define **fit scope** (train-only or OOF).

Tree boosters with **native categorical** support search splits $X\in S$; **no hand-built numeric encoding is required**. Target-type encodings can still add a **scalar** summary of $P(Y\mid X{=}c)$; validate on hold-out [7].

**Embeddings** are out of scope here; if trained with labels, the same **estimator + scope** questions apply.

---

## 6. Correlation is not redundancy

In practice, teams compute a **correlation matrix** on encoded columns, see large $|r|$, and stop—without asking what happens to **model** quality if a column is removed.

**Fact.** Pearson correlation between two **target-encoded** columns measures **linear association across rows** between the assigned level rates. It does **not** imply **conditional redundancy** for $Y$: $P(Y\mid X_1)$ and $P(Y\mid X_2)$ can differ sharply for specific level pairs even when columns correlate.

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

Naïve training target encodings yield $\mathrm{Corr}(z_1,z_2)\approx 0.72$, yet $P(Y{=}1\mid X_1{=}A)=1$ while $P(Y{=}1\mid X_2{=}M)=1/4$, and a logistic model on **both** encodings can outperform either alone on this table.

**Main conclusion (models).** **Do not drop a feature from a predictive model because it correlates with another** until you compare **models with and without** that feature (or use target-aware scores such as permutation importance). Marginal correlation is not a substitute for **held-out contribution to $Y$**.

**Pipeline experiments.** On the synthetic dataset (§9), row-wise TE correlation may fall **below** $0.7$ when many levels pool rows. The **principle** stands: use **model metrics** and §7’s checklist, not $|r|$ alone.

---

## 7. After correlations: decision checklist

Use this **after** computing pairwise correlations among encoded columns (especially target-derived).

**Do not**

- Drop a column solely because $|r|>0.9$ **without** scoring its link to $Y$.

**Do**

1. Flag pairs with high $|r|$.
2. For each pair, score **each** column against $Y$ (mutual information, permutation importance, or nested model comparison).
3. If **both** move validation metrics, **keep both** unless simplicity wins—**document** the trade-off.
4. If one is inert on hold-out, consider dropping **that** one—**document** the metric.
5. **Write the decision** with numbers (metrics, $n_c$, fold policy)—not “we removed collinear features.”

**Safer screens** all ask whether the feature **changes predictions of $Y$**, not merely whether it tracks another feature [7].

---

## 8. Target leakage: when the estimator sees the answer

**Leakage (narrow sense):** a feature value for a row **uses that row’s label** (or future labels) in a way that **cannot** happen at scoring time.

**Minimal example (three rows, one level).** Levels $\{c,c,c\}$, labels $(1,0,0)$. **Naïve** target rate for $c$ computed **including** the row gives each row a coding that **depends on its own** $Y$. **Training** loss can look excellent because the model sees a **proxy for $Y$** inside the feature; **test** performance is the honest judge—and **proper** encoding uses counts **excluding** the row (OOF/LOO) or **train-only** scopes [6].

**Impact on the model.** Leakage **inflates training metrics** (AUC-PR, accuracy) and yields **misleading feature importance**; it **does not** create information available in deployment, so the **generalisation** story breaks.

**Mechanism (naïve target encoding).** Pooling train+test (or fold) labels into $k_c$ lets the encoding **encode the answer key** for rows in the pool. The fix is **out-of-fold** or **strictly training-only** statistics for encoding maps [6].

**Low support amplifies severity.** Tiny $n_c$ means one label moves $\hat{p}_c$ sharply—self-influence is large.

**Detection smoke test.** If training AUC-PR jumps while validation barely moves after adding target encoding, **first** audit **where** $k_c$ and $n_c$ were computed.

---

## 9. Experiments

**Reproducibility.** From the **repository root**, with Python 3.10+:

```text
python -m venv .venv && source .venv/bin/activate   # or Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_all.py
```

Figures are written under `figures/` at the DPI in `config.yaml` (default 300). The generator is documented in `docs/dataset-design.md`; numeric caveats in `docs/experiments-summary.md`.

**Figure paths.** In Markdown viewers that resolve links **relative to the repository root** (e.g. GitHub file view from the repo home), use `figures/...`. When previewing **from** `article/`, use `../figures/...`. Below, paths are given **from the repository root**.

| Experiment | Main insight | Figure (repo root path) |
|------------|----------------|-------------------------|
| A | Wide intervals for tiny $n_c$; smoothing pulls rare levels toward $\bar{p}$ | `figures/exp_a_perfect_feature.png` |
| B | Naïve TE often larger train–test gap than smoothed; rankings vary by seed | `figures/exp_b_smoothing_effect.png` |
| C | High $|r|$ does not justify dropping a column without model/target-aware checks | `figures/exp_c_correlation_trap.png` |
| D | Leaky pooling inflates **training** AUC-PR vs proper OOF/train-only scope | `figures/exp_d_encoding_comparison.png` |

### Experiment A — The “perfect feature” illusion

**Setup.** Stratified split; **all** Uruguay rows in **training** so $n_c{=}5$, $k_c{=}5$ in-train. Naïve $\hat{p}_c$ and 95% Agresti–Coull intervals on train. Second panel: posterior mean for Uruguay vs prior strength $m=\alpha+\beta$.

**Figure 1** (`figures/exp_a_perfect_feature.png`). **Left:** $\hat{p}_c$ vs $n_c$ (log scale) with error bars; Uruguay annotated. **Right:** smoothed mean for Uruguay vs $m$, line at $\bar{p}$.

**Observation.** Uruguay’s interval stays wide at $\hat{p}_c{=}1$; larger $m$ shrinks toward $\bar{p}$.

**Theory link.** §§2–4.

### Experiment B — Smoothing and generalisation

**Setup.** XGBoost with numerics plus **country** as naïve target, smoothed target (`config.yaml`), or one-hot. Train/test AUC-PR and F1 at 0.5.

**Figure 2** (`figures/exp_b_smoothing_effect.png`). Test AUC-PR by encoding; train vs test AUC-PR bars.

**Observation.** Naïve target often shows a **larger** train–test gap than smoothed; exact ranking varies.

**Theory link.** §§4–5.

### Experiment C — Correlation versus redundancy

**Setup.** Naïve TE for `country` and `merchant_category` on **training only**; Pearson $r$; MI with $Y$; XGBoost with both vs one column dropped.

**Figure 3** (`figures/exp_c_correlation_trap.png`). Scatter; test AUC-PR for {both, country only, merchant only}; MI bars.

**Observation.** Do not drop on $|r|$ alone; bar charts tie claims to **model** behaviour.

**Theory link.** §§6–7.

### Experiment D — Leakage via encoding scope

**Setup.** **Leaky:** TE using **concatenated** train+test labels. **Proper:** OOF on train; test from **train-only** stats. See `scripts/experiment_d_encoding_comparison.py` for test fraction.

**Figure 4** (`figures/exp_d_encoding_comparison.png`). Train vs test AUC-PR: leaky vs proper.

**Observation.** Training AUC-PR can be **optimistic** under leakage.

**Theory link.** §8; [6].

---

## 10. A framework for encoding decisions

**Axes.** **Cardinality**, **support** ($n_c$), **model family** (linear/NN vs native categoricals).

**Decision flow (high level).**

1. List levels with **small** $n_c$; flag them for intervals, smoothing, or pooling—not raw point estimates alone.
2. Choose encoding: see §5 table + “when to use.”
3. If using **any** target-based map: define **fit scope** (train-only, OOF, or CV) **before** tuning.
4. After correlation matrices on encodings: run **§7 checklist** before dropping columns.
5. Validate on **held-out** data; watch **train vs validation** gaps.

**Heuristic bullets.**

- High cardinality + heavy tail of small $n_c$: prefer **smoothed target** or **frequency** + regularised models.
- Linear / NN: **one-hot** or **target-type** with **cross-fitting**.
- Boosted trees with native cats: encoding optional; validate added target columns.
- After high $|r|$ among encodings: **§7** before dropping.

**Compass table** (not a law—always validate on hold-out):

| Cardinality | Typical support | Model (examples) | First-line encoding | Fit scope for target-based maps |
|-------------|-----------------|------------------|---------------------|--------------------------------|
| Low | High per level | Logistic regression | One-hot or smoothed target | Train only; CV for target |
| High | Heavy tail | XGBoost (native cat) | Native + optional smoothed target | OOF / train-only |
| High | Heavy tail | Neural network | Embedding or frequency + numeric | Avoid label leakage in training |
| Any | Any | Any | High $|r|$ between encodings | §7 **before** dropping |

---

## 11. Conclusion

We framed two common gaps—**trusting extreme rates at tiny $n_c$** and **stopping at correlation**—as instances of confusing **estimates** with **truths**.

**What this article established**

- The MLE $\hat{p}_c$ and **variance** under low support; plug-in failure at boundaries.
- **Agresti–Coull** (and related) intervals to show uncertainty for **high** $\hat{p}_c$, not only 100%.
- **Beta–Binomial** smoothing as a transparent shrinkage mechanism; links to **libraries** and **pseudocode**.
- Encodings classified by **estimand** and **when to use** them.
- **Correlation $\not\Rightarrow$ redundancy** for $Y$; **model comparisons** and §7 checklist.
- **Leakage** as wrong scope, with a **minimal row-level example** and metric impact.
- Four figures tied to code; **synthetic** data as pedagogy with **production** caveats.

**Conditional takeaways**

- Small $n_c$: pair rates with **intervals** or **smoothing**; watch **overfitting**.
- High correlation between target encodings: **score against $Y$** and compare **models** before dropping.
- Target encoding: define the **sample** for $(k_c,n_c)$ as carefully as the train/test split.

**Limitations.** Quantitative claims (AUC-PR, F1, $r$) depend on `src/data.py` and `config.yaml`. Logical claims (variance, intervals, leakage, correlation vs model redundancy) do not.

**Closing line.** **Treat every encoded column as an estimate tied to a sample size and a fit scope—then validate the model you actually ship.**

---

## Appendix A: Repository file map

| Article locus | Code / docs |
|---------------|-------------|
| §§3–4, Fig. 1 | `scripts/experiment_a_perfect_feature.py`, `src/stats_utils.py` |
| §§4–5, Fig. 2 | `scripts/experiment_b_smoothing_effect.py`, `src/encoding.py`, `src/models.py` |
| §§6–7, Fig. 3 | `scripts/experiment_c_correlation_trap.py` |
| §8, Fig. 4 | `scripts/experiment_d_encoding_comparison.py` |
| Generator | `src/data.py`, `docs/dataset-design.md` |
| Numeric summary | `docs/experiments-summary.md` |
| Symbols | `article/notation.md` |
| BibTeX | `article/references.bib` |

Run all figures: `python scripts/run_all.py` from the repository root.

---

## Appendix B: Production-oriented checklist

Use alongside §7 and §10 before merging encoding changes.

- [ ] For each high-impact categorical, report **$n_c$** per level (or monitor in dashboards).
- [ ] Define **low-support** thresholds per feature; route rare levels to **Other**, smoothing, or hierarchical pooling.
- [ ] Never deploy **naïve** target statistics fit on **validation/test** labels.
- [ ] Prefer **OOF/CV** target encoding or **train-only** maps; log the policy in model cards.
- [ ] After correlation analysis on encodings, complete **§7** before dropping columns.
- [ ] Compare **validation** metrics with vs without suspect columns; watch **train–validation gap**.
- [ ] Re-validate encodings under **refresh** or **drift** (counts and rates change over time).

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
