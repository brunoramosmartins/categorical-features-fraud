"""Run all four experiments (figures → figures/*.png)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = [
    "experiment_a_perfect_feature.py",
    "experiment_b_smoothing_effect.py",
    "experiment_c_correlation_trap.py",
    "experiment_d_encoding_comparison.py",
]


def main() -> None:
    cwd = Path.cwd().resolve()
    if cwd != _ROOT:
        print(f"Note: cwd is {cwd}; experiments expect imports from {_ROOT}")
        print("Run: python scripts/run_all.py from the repository root, or: cd", _ROOT)
    os.chdir(_ROOT)
    for name in _SCRIPTS:
        path = _ROOT / "scripts" / name
        print(f"\n=== Running {name} ===")
        subprocess.run([sys.executable, str(path)], cwd=str(_ROOT), check=True)
    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
