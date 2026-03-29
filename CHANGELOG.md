# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Article third draft:** EDA-centric core (high target ratio in categoricals + which statistics validate); **general** highly-correlated predictors (numeric/encoded/mixed) with VIF, permutation importance, MI, nested models; removed “legal-style” labels; generic prose examples; `../figures/` links + §9 image embeds; GitHub math note (`n_{c}`, `\lvert r \rvert`); `docs/article-roadmap-v3.md`; README GitHub preview note. EN + pt-BR.
- **Article second draft (technical review):** generalised framing beyond the 100%/interview-only story; shorter abstract with bias/overfitting; intuition-before-formalism in §2; low-support table with 70% example and rule-of-thumb \(n_c\); §4 library bridge + pseudocode; §5 when-to-use; §6–8 strengthened for model impact and checklist formatting; §9 summary table + figure path note (root vs `article/`); §10 decision flow; §11 bullet recap + closing line; new **Appendix B** production checklist. Mirrored in `article/features-that-lie.pt-BR.md`; `docs/outline.md` updated.

### Added
- **Phase 4 (article):** full draft in `article/features-that-lie.md` (Abstract, §§1–11, Appendix A file map, References [1]–[7]); `article/notation.md` glossary; `article/references.bib`; supporting `docs/thesis.md`, `docs/outline.md`.
- README architecture diagram (Mermaid), article links, `check_env` / `verify_publication_ready` usage, development practices pointer to `CONTRIBUTING.md`.
- Phase 5 review assets: `docs/phase5-review-checklist.md`, `docs/peer-review-feedback-template.md`, `docs/phase5-figure-alt-text.md`, `notes/phase5-self-review.md`, `scripts/verify_publication_ready.py`
- Initial repository structure and project governance
- `.gitignore`, `LICENSE` (MIT), `LICENSE-TEXT` (CC BY 4.0)
- Folder skeleton: `article/`, `docs/`, `src/`, `scripts/`, `notebooks/`, `figures/`, `data/`, `notes/`
- GitHub templates: PR template, issue templates (content, code)
- `CONTRIBUTING.md` with branch, commit, and merge conventions
- `requirements.txt` with pinned dependencies
- `config.yaml` with dataset, encoding, model, and figure parameters
