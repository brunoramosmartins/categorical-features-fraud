# Contributing

Guidelines for contributing to this project.

## Branch Naming Convention

| Branch type | Pattern | Example |
|---|---|---|
| Phase development | `phase/<N>-<short-description>` | `phase/1-statistical-foundation` |
| Bug fix | `fix/<short-description>` | `fix/smoothing-formula-typo` |
| Documentation | `docs/<short-description>` | `docs/update-readme` |
| Post-publication fix | `fix/post-publication-<desc>` | `fix/post-publication-broken-link` |

**Rules:**
- Always branch from `main`.
- Never commit directly to `main`.
- Delete branches after merging.

## Commit Message Convention

This project follows [Conventional Commits](https://www.conventionalcommits.org/).

Format: `<type>(<scope>): <short description>`

| Type | When to use |
|---|---|
| `feat` | Adding new content, code, or section |
| `fix` | Correcting an error (code, math, text) |
| `docs` | Documentation-only changes |
| `refactor` | Code restructuring without behaviour change |
| `test` | Adding or modifying experiment scripts |
| `chore` | Tooling, dependencies, configuration |
| `style` | Formatting, whitespace (no logic change) |

**Examples:**
```
feat(theory): add Beta-Binomial derivation for Bayesian smoothing
feat(encoding): implement fold-based target encoding in src/encoding.py
fix(experiment-a): correct Agresti-Coull CI computation for Uruguay
docs(readme): add reproduction instructions
chore(deps): pin category-encoders to 2.6.3
```

## Merge Strategy

All merges to `main` use **squash and merge** via Pull Request.

Each phase produces one PR with a title following the convention:
`[Phase N] Short description`

## Pull Request Rules

1. Every PR must reference the milestone it closes.
2. Every PR must list the issues it closes.
3. Every PR must pass the checklist in the PR template.
4. Code PRs must include verification that `python scripts/run_all.py` runs without errors.
