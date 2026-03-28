# Phase 5 — Review, polish & pre-publication checklists

Use this document while executing **Phase 5** of the roadmap (`phase/5-review-polish`). Tick boxes in your working copy or in the PR description.

**Inputs:** first complete draft in `article/features-that-lie.md` (Phase 4), figures in `figures/`, code on `main` or release branch.

---

## 5.1 — Self-review pass

- [ ] Read the entire article **aloud**. Mark unclear or overly long sentences.
- [ ] Every mathematical claim is **derived**, **cited**, or explicitly marked as a **heuristic**.
- [ ] The **thesis** is clearly answered in the **conclusion**.
- [ ] The **narrative thread** is coherent: recurring practice gaps (extreme rates at low \(n_c\); correlation without decision rule) → shared root cause → checklists and experiments.
- [ ] Thesis uses **softened** language (“more likely evidence of …”), not absolute claims.
- [ ] Scan for **absolute** statements; replace with **conditional** ones where appropriate.
- [ ] **Terminology** is consistent: e.g. “high-variance estimator”, “low-support regime”.

---

## 5.2 — Technical accuracy review

- [ ] Re-derive **every** formula in the article (pencil/paper or separate notes).
- [ ] Encoding definitions match **scikit-learn** and **category_encoders** docs (links in references).
- [ ] **Uruguay** numerical example recomputed by hand (Agresti–Coull / interval width).
- [ ] **Bayesian smoothing** derivation checked against Bishop (2006) or Murphy (2012).
- [ ] **Correlation counterexample** (Phase 1 notes): recompute correlation and conditional rates.
- [ ] `python scripts/run_all.py` succeeds in a **clean** virtual environment (`pip install -r requirements.txt`).

---

## 5.3 — Peer review

- [ ] Draft shared with **at least one** technical reader.
- [ ] Written feedback collected; decisions recorded (see `docs/peer-review-feedback-template.md`).

---

## 5.4 — Language & readability

- [ ] Grammar and **style** pass (consistent dialect: en-GB vs en-US).
- [ ] **Section transitions:** each section ends by motivating the next.
- [ ] **Framing** stays professional: context/problem/risk in the introduction; avoid over-reliance on a single anecdote unless intentional.

---

## 5.5 — Visual polish & accessibility

- [ ] All figures regenerated at **final** resolution (`config.yaml` → `figures.dpi`, default 300).
- [ ] **Consistent** colour palette and fonts across figures (adjust `src/plotting.py` / experiment scripts if needed).
- [ ] **Alt-text** drafted for every figure (see `docs/phase5-figure-alt-text.md`); embedded in portfolio HTML or adjacent doc as required.
- [ ] **Portfolio** build: preview exported HTML (Markdown → Python pipeline in the **portfolio repo**).

---

## 5.6 — Length & scope

- [ ] Line count for `article/features-that-lie.md`: target **~500** lines; if **> 600**, plan cuts.
- [ ] **Leakage** subsection stays **brief** (subsection weight, not a full chapter).
- [ ] **Encoding landscape** lighter than **estimation / smoothing** sections.

---

## 5.7 — Final pre-publication checklist

- [ ] Thesis answered clearly in **conclusion**.
- [ ] Code runs **end-to-end** from clean environment.
- [ ] All **figures** present and referenced correctly in the article.
- [ ] All **links** in article and README work (internal paths + external URLs).
- [ ] `LICENSE` (code) and `LICENSE-TEXT` (article) present and cited in README.
- [ ] README documents **reproduction** (`scripts/run_all.py` or equivalent).
- [ ] No **secrets**, internal URLs, or sensitive data in the repo.

---

## Automation helper

Run from repository root:

```bash
python scripts/verify_publication_ready.py
python scripts/verify_publication_ready.py --skip-experiments   # faster: no run_all
```

This does **not** replace the checklists above; it catches common mechanical gaps.
