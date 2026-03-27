#!/usr/bin/env bash
# Bootstrap GitHub labels, milestones, and issues for categorical-features-fraud.
# Run in WSL/Linux from repo root (or anywhere with gh authenticated):
#
#   chmod +x scripts/gh_bootstrap_project.sh
#   cd /path/to/categorical-features-fraud
#   gh auth login   # if needed
#   ./scripts/gh_bootstrap_project.sh
#
# Requires: gh CLI, bash. Uses gh api for milestones (works on older gh versions).
# Labels/milestones: safe to re-run (create or update / skip existing milestones).
# Issues: NOT idempotent — each run creates new issues. Run once per repository.
#   Labels + milestones only: BOOTSTRAP_ISSUES=0 ./scripts/gh_bootstrap_project.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v gh >/dev/null 2>&1; then
  echo "Install GitHub CLI: https://cli.github.com/"
  exit 1
fi

REPO="${GH_REPO:-$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)}"
if [[ -z "${REPO}" || "${REPO}" == "null" ]]; then
  echo "Set remote or export GH_REPO=owner/repo (e.g. GH_REPO=bruno/categorical-features-fraud)"
  exit 1
fi

echo "Using repository: ${REPO}"

label_upsert() {
  local name="$1" color="$2" desc="$3"
  if ! gh label create "$name" --color "$color" --description "$desc" --repo "$REPO" 2>/dev/null; then
    gh label edit "$name" --color "$color" --description "$desc" --repo "$REPO" 2>/dev/null || true
  fi
}

echo "==> Labels"
label_upsert "setup" "0075ca" "Repository and environment configuration"
label_upsert "research" "e4e669" "Literature review, theory, references"
label_upsert "writing" "d93f0b" "Article writing and narrative"
label_upsert "code" "0e8a16" "Experiment code and scripts"
label_upsert "experiment" "1d76db" "Running and documenting experiments"
label_upsert "review" "cc317c" "Review, proofreading, polish"
label_upsert "publishing" "5319e7" "Publishing and promotion tasks"
label_upsert "bug" "ee0701" "Something is incorrect or broken"
label_upsert "documentation" "0075ca" "README, CONTRIBUTING, inline docs"
for p in 0 1 2 3 4 5 6; do
  label_upsert "phase-${p}" "ededed" "Phase ${p} tracking"
done

milestone_create() {
  local title="$1"
  local desc="$2"
  local found
  found=$(gh api "repos/${REPO}/milestones" --paginate --jq ".[] | select(.title==\"${title}\") | .number" 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    echo "  skip (exists): $title"
    return 0
  fi
  gh api -X POST "repos/${REPO}/milestones" -f title="$title" -f description="$desc" >/dev/null
  echo "  created: $title"
}

echo "==> Milestones"
milestone_create "Phase 0: Foundation & Setup" "Repo structure, governance, thesis, dataset design. Tag v0.0-init."
milestone_create "Phase 1: Statistical Foundation" "MLE, variance, smoothing, correlation counterexample. Tag v0.1-theory."
milestone_create "Phase 2: Encoding Landscape & Consequences" "Encoding strategies, leakage as consequence. Tag v0.2-encoding."
milestone_create "Phase 3: Experiments & Code" "Four experiments, run_all.py, reproducibility. Release v0.3.0-experiments."
milestone_create "Phase 4: Article Writing" "First complete draft (~500 lines). Release v0.4.0-draft."
milestone_create "Phase 5: Review & Polish" "Publication-ready. Release v1.0.0."
milestone_create "Phase 6: Publishing & Promotion" "Portfolio repo (Markdown→HTML), project card, LinkedIn, maintenance."

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if [[ "${BOOTSTRAP_ISSUES:-1}" != "1" ]]; then
  echo "BOOTSTRAP_ISSUES=0 — skipping issue creation."
  echo "Done (labels + milestones only)."
  exit 0
fi

create_issue() {
  local title="$1"
  local body_file="$2"
  shift 2
  gh issue create --repo "$REPO" --title "$title" --body-file "$body_file" "$@"
}

echo "==> Issues (Phase 0–3 full text; 4–6 from roadmap tables)"

# --- Phase 0 ---
cat >"$TMP/i01.md" <<'EOF'
## Context
The repository needs a complete initial structure before any content work begins.

## Tasks
- [ ] Create repository on GitHub
- [ ] Add `.gitignore` (Python), `LICENSE` (MIT), `LICENSE-TEXT` (CC BY 4.0)
- [ ] Create full folder structure per Appendix C (roadmap)
- [ ] Write `README.md` skeleton: project title, one-paragraph summary, "How to reproduce" placeholder, license badges

## Definition of done
- [ ] `git clone` + `ls -R` shows the complete structure
- [ ] README renders correctly on GitHub

## References
- Appendix C of project roadmap
EOF
create_issue "[SETUP] Initialize repository structure and base files" "$TMP/i01.md" \
  --label "setup" --label "phase-0" --milestone "Phase 0: Foundation & Setup"

cat >"$TMP/i02.md" <<'EOF'
## Context
Full project governance must be established before content work begins.

## Tasks
- [ ] Create all 7 milestones (Phase 0 through Phase 6) with descriptions
- [ ] Create all labels per Appendix D labels table
- [ ] Create issue templates: `content.md` and `code.md` per Appendix D
- [ ] Create `.github/pull_request_template.md` per Appendix D
- [ ] Write `CONTRIBUTING.md` with branch naming, commit conventions, merge strategy

## Definition of done
- [ ] All milestones visible in GitHub UI
- [ ] All labels created with correct colours
- [ ] Both issue templates functional
- [ ] PR template appears when opening a new PR

## References
- Appendix D of project roadmap
EOF
create_issue "[SETUP] Create milestones, labels, issue templates, and PR template" "$TMP/i02.md" \
  --label "setup" --label "phase-0" --milestone "Phase 0: Foundation & Setup"

cat >"$TMP/i03.md" <<'EOF'
## Context
The thesis must be precise enough to be testable by the experiments. The scope must be tight enough to produce an article of comparable length to the ~530-line Article 1 standard.

## Tasks
- [ ] Write the final thesis sentence — statistical, not categorical
- [ ] List explicit out-of-scope items
- [ ] Write reader persona paragraph
- [ ] Draft "promise to the reader" (5 bullets)
- [ ] Write the narrative thread: two interview gaps → shared root cause → resolution
- [ ] Draft the article skeleton (~11 sections) with 2-line descriptions
- [ ] Verify every section connects to the central axis (estimators under uncertainty)
- [ ] Verify leakage is subsection-weight, not a full section

## Definition of done
- [ ] `docs/thesis.md` and `docs/outline.md` exist
- [ ] Thesis is statistical ("more likely evidence of X") not categorical ("is evidence of X")
- [ ] Central axis is named explicitly: "high-variance estimators under low support"
- [ ] Article targets ~11 sections and ~500 lines of final markdown

## References
- LLM feedback on scope tightening and thesis softening
EOF
create_issue "[FOUNDATION] Write final thesis, scope, and article skeleton" "$TMP/i03.md" \
  --label "writing" --label "phase-0" --milestone "Phase 0: Foundation & Setup"

cat >"$TMP/i04.md" <<'EOF'
## Context
The original case used a private dataset. The article needs a synthetic dataset reproducing key phenomena: Uruguay (low support, extreme rate), correlated-but-non-redundant encoded features, and conditions where smoothing improves generalisation.

## Tasks
- [ ] Define all categorical features with exact parameters per country/category
- [ ] Verify (quick script) that the four target phenomena are reproducible:
      (1) Uruguay: 100% rate, CI spanning ~[0.45, 1.0]
      (2) Corr(country_encoded, merchant_encoded) > 0.7
      (3) P(Y|country=c) ≠ P(Y|merchant=m) for specific pairs
      (4) Smoothed model outperforms naïve on test set for rare categories
- [ ] Document in `docs/dataset-design.md`

## Definition of done
- [ ] `docs/dataset-design.md` exists with full specification
- [ ] Quick prototype confirms all four phenomena
- [ ] Parameters in `config.yaml`

## References
- Article 1 synthetic dataset (for design approach reference)
EOF
create_issue "[FOUNDATION] Design synthetic dataset" "$TMP/i04.md" \
  --label "research" --label "phase-0" --milestone "Phase 0: Foundation & Setup"

cat >"$TMP/i05.md" <<'EOF'
## Context
Reproducibility requires pinned dependencies and centralised configuration.

## Tasks
- [ ] Write `requirements.txt` with pinned versions
- [ ] Write `config.yaml` with dataset, encoding, models, figures sections
- [ ] Verify: `pip install -r requirements.txt` succeeds in clean env

## Definition of done
- [ ] `requirements.txt` and `config.yaml` exist
- [ ] Clean install verified

## References
- Libraries: scikit-learn, numpy, pandas, matplotlib, seaborn, category_encoders, xgboost, scipy, pyyaml
EOF
create_issue "[SETUP] Python environment and config.yaml" "$TMP/i05.md" \
  --label "setup" --label "phase-0" --milestone "Phase 0: Foundation & Setup"

# --- Phase 1 ---
cat >"$TMP/i06.md" <<'EOF'
## Context
Every category-level statistic used in encoding is an estimator. This issue establishes what "estimator" means, derives its variance, and names the problem: high-variance estimator under low-support regime.

## Tasks
- [ ] Define setup: categorical feature X, level c, counts k_c and n_c
- [ ] State the MLE: p̂ = k/n
- [ ] Derive Var(p̂) = p(1-p)/n
- [ ] Name the regime: "high-variance estimator under low support"
- [ ] State the failure mode when p̂ = 1.0 (plug-in variance is zero)

## Definition of done
- [ ] Derivation complete and reviewed
- [ ] Terminology ("low-support regime", "high-variance estimator") used consistently
- [ ] Notes in `notes/phase1-theory.md`

## References
- Casella & Berger (2002), Statistical Inference, Ch. 7
- Agresti & Coull (1998)
EOF
create_issue "[THEORY] MLE estimator and variance for category-level target rates" "$TMP/i06.md" \
  --label "research" --label "phase-1" --milestone "Phase 1: Statistical Foundation"

cat >"$TMP/i07.md" <<'EOF'
## Context
Uruguay (k≈5, n≈5, p̂=1.0) is the first narrative anchor. This issue formalises why the statistic is unreliable, uses the Agresti-Coull interval, and quantifies the uncertainty with explicit numbers.

## Tasks
- [ ] Apply MLE framework to Uruguay
- [ ] Introduce Agresti-Coull adjustment: p̃ = (k+2)/(n+4)
- [ ] Compute 95% CI ≈ [0.45, 0.94]
- [ ] State punchline: "more likely evidence of insufficient data"
- [ ] Contrast with Brazil (n≈40,000): tight CI
- [ ] Include micro-quantification: "approximately five transactions"

## Definition of done
- [ ] CIs computed for Uruguay and Brazil
- [ ] Punchline uses softened language (statistical, not categorical)
- [ ] Notes in `notes/phase1-theory.md`

## References
- Agresti & Coull (1998)
- Brown, Cai & DasGupta (2001)
EOF
create_issue "[THEORY] The Uruguay problem — numerical deconstruction with quantification" "$TMP/i07.md" \
  --label "research" --label "phase-1" --milestone "Phase 1: Statistical Foundation"

cat >"$TMP/i08.md" <<'EOF'
## Context
The mathematical heart of the article. Bayesian smoothing regularises the MLE towards the global mean, with the amount controlled by prior strength relative to sample size.

## Tasks
- [ ] Define the Beta-Binomial conjugate model
- [ ] Derive the posterior mean
- [ ] Show as weighted average of MLE and prior mean
- [ ] Derive weight function w_c = n_c / (n_c + α + β)
- [ ] Apply to Uruguay: shrinkage from 1.0 towards global mean
- [ ] Apply to Brazil: minimal effect
- [ ] Connect to category_encoders smoothing parameter

## Definition of done
- [ ] Full derivation complete
- [ ] Numerical examples match expectations
- [ ] Notes in `notes/phase1-theory.md`

## References
- Bishop (2006), PRML, Ch. 2.1
- Murphy (2012), MLAPP, Ch. 3.3
EOF
create_issue "[THEORY] Bayesian smoothing — Beta-Binomial conjugate model" "$TMP/i08.md" \
  --label "research" --label "phase-1" --milestone "Phase 1: Statistical Foundation"

cat >"$TMP/i09.md" <<'EOF'
## Context
The second narrative anchor: high correlation was observed between features during the case, but no framework existed for deciding what to do. This issue constructs the formal counterexample proving that correlation between encoded features does not imply predictive redundancy.

## Tasks
- [ ] Construct dataset: 8+ rows, 2 categoricals, 1 binary target
- [ ] Compute target encoding for both; show Corr > 0.7
- [ ] Show P(Y|X1=a) ≠ P(Y|X2=b) for specific pairs
- [ ] Train models: X1 alone, X2 alone, both → show both outperforms either
- [ ] State conclusion: marginal correlation ≠ conditional redundancy

## Definition of done
- [ ] Counterexample fully constructed with explicit numbers
- [ ] All three model comparisons computed
- [ ] Notes in `notes/phase1-theory.md`

## References
- LLM feedback: "Contraexemplo formal de correlação"
EOF
create_issue "[THEORY] Formal counterexample: correlation ≠ redundancy" "$TMP/i09.md" \
  --label "research" --label "phase-1" --milestone "Phase 1: Statistical Foundation"

cat >"$TMP/i10.md" <<'EOF'
## Context
This issue completes the correlation gap from the interview by providing the decision framework that was missing: how to move from "I see high correlation" to "here is what I do and why."

## Tasks
- [ ] Analyse: correlation threshold (unsafe), mutual information (safe), permutation importance (safe)
- [ ] Write the 5-step decision heuristic
- [ ] State explicitly: "this answers the gap — the analysis stopped at observation; this framework completes it"

## Definition of done
- [ ] Decision heuristic documented as a numbered checklist
- [ ] Each method classified as safe/unsafe/conditional
- [ ] Notes in `notes/phase1-theory.md`

## References
- Scikit-learn: mutual_info_classif, permutation_importance
EOF
create_issue "[THEORY] Feature selection under encoding — when to drop, when to keep" "$TMP/i10.md" \
  --label "research" --label "phase-1" --milestone "Phase 1: Statistical Foundation"

cat >"$TMP/i11.md" <<'EOF'
## Context
After completing theoretical work, draft rough article sections. Focus on claim → derivation → consequence structure.

## Tasks
- [ ] Draft Section 2 (MLE and estimator framing)
- [ ] Draft Section 3 (Uruguay deconstruction)
- [ ] Draft Section 4 (Bayesian smoothing)
- [ ] Draft Section 6 (Correlation ≠ redundancy)
- [ ] Draft Section 7 (When to drop, when to keep)

## Definition of done
- [ ] Rough drafts in `notes/` with all math
- [ ] Each section follows: definition → derivation → consequence
- [ ] No section relies on intuition without a corresponding formula

## References
- `notes/phase1-theory.md`
EOF
create_issue "[WRITING] Draft rough theory sections for Phase 1" "$TMP/i11.md" \
  --label "writing" --label "phase-1" --milestone "Phase 1: Statistical Foundation"

# --- Phase 2 ---
cat >"$TMP/i12.md" <<'EOF'
## Context
Each encoding strategy maps categories to numbers. The unified question is: what does each number estimate? The answer determines when the encoding is reliable and when it is not.

## Tasks
- [ ] Define each encoding with formula and "what it estimates" label
- [ ] Build the encoding comparison table
- [ ] State advantages, risks, and cardinality handling for each

## Definition of done
- [ ] Comparison table complete with all four strategies
- [ ] Each encoding connected to Phase 1 estimator framework
- [ ] Notes in `notes/phase2-encoding.md`

## References
- Micci-Barreca (2001)
- scikit-learn and category_encoders documentation
EOF
create_issue "[THEORY] Encoding strategies — what each estimates" "$TMP/i12.md" \
  --label "research" --label "phase-2" --milestone "Phase 2: Encoding Landscape & Consequences"

cat >"$TMP/i13.md" <<'EOF'
## Context
The case used XGBoost, which handles categoricals natively. The article must address when encoding is necessary vs optional.

## Tasks
- [ ] Define what native categorical handling means algorithmically
- [ ] Construct decision matrix: model type × encoding → recommendation
- [ ] State when encoding adds value even when not required

## Definition of done
- [ ] Decision matrix complete
- [ ] Notes in `notes/phase2-encoding.md`

## References
- XGBoost categorical feature documentation
EOF
create_issue "[THEORY] Encoding vs native categorical handling" "$TMP/i13.md" \
  --label "research" --label "phase-2" --milestone "Phase 2: Encoding Landscape & Consequences"

cat >"$TMP/i14.md" <<'EOF'
## Context
Leakage is a direct consequence of computing the target encoding estimator on the wrong data. This is NOT a standalone leakage taxonomy — it appears here because it follows naturally from the estimation framework.

## Tasks
- [ ] Define leakage in 1 paragraph
- [ ] Show the 3-step mechanism
- [ ] Connect severity to low-support problem (small n_c → bigger leak)
- [ ] State the fix: fold-based or LOO encoding
- [ ] State one detection test: train-vs-validation AUC-PR gap
- [ ] Keep this brief — ~1 page of notes, not a full section

## Definition of done
- [ ] Mechanism documented concisely
- [ ] Connected to Phase 1 low-support framework
- [ ] Notes in `notes/phase2-encoding.md`

## References
- Kaufman et al. (2012)
- scikit-learn TargetEncoder cross-fitting docs
EOF
create_issue "[THEORY] Target leakage as a consequence of improper encoding" "$TMP/i14.md" \
  --label "research" --label "phase-2" --milestone "Phase 2: Encoding Landscape & Consequences"

cat >"$TMP/i15.md" <<'EOF'
## Context
Draft the article sections for the encoding landscape and leakage. These are shorter sections — the depth is in Phase 1.

## Tasks
- [ ] Draft Section 5 (encoding landscape)
- [ ] Draft Section 8 (leakage as consequence — brief)
- [ ] Draft the encoding comparison table as a figure

## Definition of done
- [ ] Rough drafts in `notes/`
- [ ] Encoding section is lighter in math than Phase 1 sections
- [ ] Leakage section is ≤ 1 page (subsection weight)

## References
- `notes/phase2-encoding.md`
EOF
create_issue "[WRITING] Draft rough encoding and leakage sections" "$TMP/i15.md" \
  --label "writing" --label "phase-2" --milestone "Phase 2: Encoding Landscape & Consequences"

# --- Phase 3 ---
cat >"$TMP/i16.md" <<'EOF'
## Context
The synthetic dataset must reproduce the key phenomena designed in Phase 0.

## Tasks
- [ ] Implement data generation per `docs/dataset-design.md`
- [ ] Validate: Uruguay ≈ 5 obs, 100% fraud; Brazil ≈ 40k, ~0.4%
- [ ] Validate: Corr(country_encoded, merchant_encoded) > 0.7
- [ ] Stratified train/test split
- [ ] All params from `config.yaml`

## Expected output
- `src/data.py` with `generate_dataset()` and `load_and_split()` functions
- Updated `docs/dataset-design.md` with actual statistics

## Definition of done
- [ ] Script runs without errors
- [ ] All four phenomena validated
- [ ] Seed produces identical output across runs
EOF
create_issue "[CODE] Synthetic dataset generation (src/data.py)" "$TMP/i16.md" \
  --label "code" --label "phase-3" --milestone "Phase 3: Experiments & Code"

cat >"$TMP/i17.md" <<'EOF'
## Context
All encoding strategies must be implemented as clean functions that read from config and return consistently-named columns.

## Tasks
- [ ] Implement: one_hot, target_naive, target_leaky, smoothed_target, fold_target
- [ ] Column naming: `{feature}_{encoding_type}`
- [ ] All read from config.yaml

## Expected output
- `src/encoding.py` with 5 encoding functions

## Definition of done
- [ ] All functions work on the synthetic dataset
- [ ] Output columns correctly named
EOF
create_issue "[CODE] Encoding utilities (src/encoding.py)" "$TMP/i17.md" \
  --label "code" --label "phase-3" --milestone "Phase 3: Experiments & Code"

cat >"$TMP/i18.md" <<'EOF'
## Context
Experiment A is the empirical anchor for the Uruguay narrative and the low-support estimation theory (Phase 1, tasks 1.1–1.3).

## Theoretical claim being tested
A category with 100% target rate and ~5 observations produces a high-variance estimate. Bayesian smoothing collapses it towards the global mean, with the effect proportional to prior strength.

## Tasks
- [ ] Target-encode country (naïve)
- [ ] Compute Agresti-Coull 95% CI for each country
- [ ] Smooth with α+β ∈ {10, 100, 1000}
- [ ] Generate plots: (a) encoded value vs sample size, (b) smoothed estimate vs prior strength

## Expected output
- `figures/exp_a_perfect_feature.png` (2-panel figure)
- Results documented in `docs/experiments-summary.md`

## Definition of done
- [ ] Uruguay CI ≈ [0.45, 1.0]
- [ ] Smoothed estimate collapses from 1.0 → near global mean
- [ ] Well-supported categories barely affected
- [ ] Figures at 300dpi
EOF
create_issue "[EXPERIMENT] Exp A: The perfect feature illusion" "$TMP/i18.md" \
  --label "experiment" --label "phase-3" --milestone "Phase 3: Experiments & Code"

cat >"$TMP/i19.md" <<'EOF'
## Theoretical claim being tested
Bayesian smoothing reduces overfitting on rare-category estimates, improving test-set performance (Phase 1, task 1.3).

## Tasks
- [ ] Train XGBoost with: naïve TE, smoothed TE, one-hot encoding of country
- [ ] Evaluate: AUC-PR on train and test
- [ ] Compute train-test gaps
- [ ] Generate plots: (a) test AUC-PR bar chart, (b) train vs test gap

## Expected output
- `figures/exp_b_smoothing_effect.png`

## Definition of done
- [ ] Naïve TE shows largest train-test gap
- [ ] Smoothed TE has best test performance
- [ ] Figures at 300dpi
EOF
create_issue "[EXPERIMENT] Exp B: Smoothing effect on model performance" "$TMP/i19.md" \
  --label "experiment" --label "phase-3" --milestone "Phase 3: Experiments & Code"

cat >"$TMP/i20.md" <<'EOF'
## Theoretical claim being tested
High correlation between target-encoded features does not imply predictive redundancy. Removing either feature reduces performance (Phase 1, tasks 1.4–1.5).

## Tasks
- [ ] Target-encode country and merchant_category
- [ ] Compute Pearson correlation
- [ ] Train: both features, country only, merchant only
- [ ] Compute MI of each with Y
- [ ] Generate plots: (a) scatter with corr annotation, (b) performance bar chart, (c) MI comparison

## Expected output
- `figures/exp_c_correlation_trap.png`

## Definition of done
- [ ] Correlation > 0.7 confirmed
- [ ] Both-feature model outperforms single-feature models
- [ ] MI confirms non-redundancy
- [ ] Figures at 300dpi
EOF
create_issue "[EXPERIMENT] Exp C: The correlation trap" "$TMP/i20.md" \
  --label "experiment" --label "phase-3" --milestone "Phase 3: Experiments & Code"

cat >"$TMP/i21.md" <<'EOF'
## Theoretical claim being tested
Computing target encoding on the full dataset inflates training performance. The inflation is detectable through the train-validation gap (Phase 2, task 2.3).

## Tasks
- [ ] Leaky pipeline: encode on full data
- [ ] Proper pipeline: fold-based encoding on train only
- [ ] Train same XGBoost model with each
- [ ] Compute train and test AUC-PR
- [ ] Generate plot: grouped bar chart with gap annotation

## Expected output
- `figures/exp_d_encoding_comparison.png`

## Definition of done
- [ ] Leaky: higher training AUC-PR, larger gap
- [ ] Proper: smaller gap, similar or better test AUC-PR
- [ ] Figures at 300dpi
EOF
create_issue "[EXPERIMENT] Exp D: Leakage through naïve encoding" "$TMP/i21.md" \
  --label "experiment" --label "phase-3" --milestone "Phase 3: Experiments & Code"

cat >"$TMP/i22.md" <<'EOF'
## Tasks
- [ ] Write `scripts/run_all.py`: data → encode → train → evaluate → plot
- [ ] Verify: fresh env → install → run → all figures regenerated
- [ ] Document in README: "How to reproduce"

## Definition of done
- [ ] Single command produces all figures
- [ ] README updated
EOF
create_issue "[CODE] run_all.py and reproducibility verification" "$TMP/i22.md" \
  --label "code" --label "phase-3" --milestone "Phase 3: Experiments & Code"

# --- Phase 4 (writing) ---
phase4_body() {
  cat <<EOF
## Context
Phase 4 article writing. Section-specific issue from roadmap.

## Definition of done
- [ ] Claim → derivation → consequence structure where applicable
- [ ] Formulas present and correct
- [ ] Notation matches \`article/notation.md\`
- [ ] Section length proportional to its role

## References
- \`docs/outline.md\`, \`notes/phase1-theory.md\`, \`notes/phase2-encoding.md\`
EOF
}

phase4_body >"$TMP/p4.md"
create_issue "[WRITING] Establish notation and writing conventions" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: A category-level statistic is an estimator" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: The low-support problem — Uruguay deconstruction" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: Bayesian smoothing — the principled fix" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: The encoding landscape" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: Correlation is not redundancy" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: When to drop, when to keep" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: Target leakage" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: Experiments section" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: Decision framework" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: Introduction" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Write: Conclusion and abstract" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"
create_issue "[WRITING] Integrate figures and complete references" "$TMP/p4.md" \
  --label "writing" --label "phase-4" --milestone "Phase 4: Article Writing"

# --- Phase 5 ---
phase5_body() {
  cat <<'EOF'
## Context
Phase 5 — review, polish, pre-publication. See roadmap task list 5.1–5.7.

## Definition of done
- [ ] Criteria in roadmap subsection satisfied for this review slice
- [ ] CHANGELOG updated if release-ready
EOF
}
phase5_body >"$TMP/p5.md"
create_issue "[REVIEW] Self-review: clarity, claims, narrative thread, and thesis language" "$TMP/p5.md" \
  --label "review" --label "phase-5" --milestone "Phase 5: Review & Polish"
create_issue "[REVIEW] Technical accuracy: re-derive all formulas" "$TMP/p5.md" \
  --label "review" --label "phase-5" --milestone "Phase 5: Review & Polish"
create_issue "[REVIEW] Code reproducibility final check" "$TMP/p5.md" \
  --label "review" --label "phase-5" --milestone "Phase 5: Review & Polish"
create_issue "[REVIEW] Peer review round" "$TMP/p5.md" \
  --label "review" --label "phase-5" --milestone "Phase 5: Review & Polish"
create_issue "[REVIEW] Language and readability pass" "$TMP/p5.md" \
  --label "review" --label "phase-5" --milestone "Phase 5: Review & Polish"
create_issue "[REVIEW] Visual polish and alt-text" "$TMP/p5.md" \
  --label "review" --label "phase-5" --milestone "Phase 5: Review & Polish"
create_issue "[REVIEW] Length and scope check (~500 lines target)" "$TMP/p5.md" \
  --label "review" --label "phase-5" --milestone "Phase 5: Review & Polish"
create_issue "[REVIEW] Final pre-publication checklist" "$TMP/p5.md" \
  --label "review" --label "phase-5" --milestone "Phase 5: Review & Polish"

# --- Phase 6 ---
phase6_body() {
  cat <<'EOF'
## Context
Phase 6 — publishing and promotion. See roadmap Phase 6 tasks.

## Definition of done
- [ ] Deliverable in roadmap completed; live article linked from README via portfolio URL when applicable
EOF
}
phase6_body >"$TMP/p6.md"
create_issue "[PUBLISH] Publish article via portfolio repo (Markdown → Python → HTML)" "$TMP/p6.md" \
  --label "publishing" --label "phase-6" --milestone "Phase 6: Publishing & Promotion"
create_issue "[PUBLISH] Add project card to portfolio site" "$TMP/p6.md" \
  --label "publishing" --label "phase-6" --milestone "Phase 6: Publishing & Promotion"
create_issue "[PUBLISH] Register DOI via Zenodo" "$TMP/p6.md" \
  --label "publishing" --label "phase-6" --milestone "Phase 6: Publishing & Promotion"
create_issue "[PUBLISH] Write and publish LinkedIn post" "$TMP/p6.md" \
  --label "publishing" --label "phase-6" --milestone "Phase 6: Publishing & Promotion"
create_issue "[PUBLISH] Post-publication maintenance plan" "$TMP/p6.md" \
  --label "publishing" --label "phase-6" --milestone "Phase 6: Publishing & Promotion"

echo "Done. Open: https://github.com/${REPO}/issues"
