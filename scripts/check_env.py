"""Verify imports and repo layout before running experiments.

Usage (from repository root):
    python scripts/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "yaml",
    "scipy",
    "xgboost",
]


def main() -> int:
    ok = True
    for name in REQUIRED:
        try:
            __import__(name if name != "yaml" else "yaml", fromlist=["_"])
        except ImportError:
            print(f"MISSING: {name}  -> pip install -r requirements.txt")
            ok = False
        else:
            print(f"ok: {name}")

    cfg_path = ROOT / "config.yaml"
    if not cfg_path.is_file():
        print(f"MISSING: {cfg_path}")
        ok = False
    else:
        print(f"ok: config.yaml")

    try:
        from src.config_utils import load_config

        load_config()
        print("ok: src.config_utils.load_config()")
    except Exception as e:
        print(f"FAIL: load_config — {e}")
        ok = False

    if not ok:
        print("\nRun from repo root: cd categorical-features-fraud")
        print("Install deps: pip install -r requirements.txt")
        return 1
    print("\nEnvironment OK. Next: python scripts/run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
