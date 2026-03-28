"""Mechanical checks for Phase 5 pre-publication (roadmap 5.7 helper).

Usage (repository root):
    python scripts/verify_publication_ready.py
    python scripts/verify_publication_ready.py --skip-experiments

Does not replace human review; see docs/phase5-review-checklist.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ARTICLE = _ROOT / "article" / "features-that-lie.md"
_EXPECTED_FIGURES = [
    "exp_a_perfect_feature.png",
    "exp_b_smoothing_effect.png",
    "exp_c_correlation_trap.png",
    "exp_d_encoding_comparison.png",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-experiments", action="store_true", help="Do not run scripts/run_all.py")
    p.add_argument(
        "--article-lines-warn",
        type=int,
        default=900,
        help="Warn if article exceeds this many lines (very long draft)",
    )
    p.add_argument(
        "--article-lines-soft-min",
        type=int,
        default=280,
        help="Warn if article has fewer lines than this (roadmap aspirational ~500)",
    )
    args = p.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    for name in ("LICENSE", "LICENSE-TEXT"):
        f = _ROOT / name
        if not f.is_file():
            errors.append(f"Missing {f.relative_to(_ROOT)}")

    readme = _ROOT / "README.md"
    if readme.is_file():
        text = _read_text(readme).lower()
        if "run_all" not in text and "reproduce" not in text:
            warnings.append("README: consider explicit 'reproduce' / run_all.py instructions")
    else:
        errors.append("Missing README.md")

    if not _ARTICLE.is_file():
        errors.append(f"Missing {_ARTICLE.relative_to(_ROOT)}")
    else:
        lines = _read_text(_ARTICLE).splitlines()
        n = len(lines)
        print(f"Article lines: {n} ({_ARTICLE.name})")
        if n < 50:
            warnings.append("Article is very short — Phase 4 draft may not be merged yet")
        elif n < args.article_lines_soft_min:
            warnings.append(
                f"Article has {n} lines (soft minimum {args.article_lines_soft_min}); "
                "roadmap aspirational length was ~500 — optional expansion"
            )
        if n > args.article_lines_warn:
            warnings.append(f"Article exceeds {args.article_lines_warn} lines — check length for target venue")

    fig_dir = _ROOT / "figures"
    for fn in _EXPECTED_FIGURES:
        fp = fig_dir / fn
        if not fp.is_file():
            warnings.append(f"Missing figure: figures/{fn} (run scripts/run_all.py)")

    if not args.skip_experiments:
        print("Running scripts/run_all.py …")
        try:
            subprocess.run(
                [sys.executable, str(_ROOT / "scripts" / "run_all.py")],
                cwd=str(_ROOT),
                check=True,
                timeout=600,
            )
            print("run_all.py: OK")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            errors.append(f"run_all.py failed: {e}")

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    if errors:
        print("\nFix errors above, then re-run.")
        return 1
    print("\nMechanical checks passed (see docs/phase5-review-checklist.md for full Phase 5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
