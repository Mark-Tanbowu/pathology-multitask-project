"""Simple helper to run multitask vs single-task ablations with consistent overrides."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


EXPERIMENTS = [
    {"name": "multitask", "overrides": []},
    {"name": "seg_only", "overrides": ["tasks.enable_cls=false"]},
    {"name": "cls_only", "overrides": ["tasks.enable_seg=false"]},
]


def run_command(args: List[str], dry_run: bool) -> int:
    cmd = [sys.executable, "-m", "src.engine.train", *args]
    print(f"[ABLT] Running: {' '.join(cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multitask vs single-task ablations.")
    parser.add_argument(
        "--extra-overrides",
        nargs="*",
        default=[],
        help="Additional Hydra overrides applied to every run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the commands without executing them.",
    )
    args = parser.parse_args()

    for exp in EXPERIMENTS:
        overrides = exp["overrides"] + args.extra_overrides
        ret = run_command(overrides, args.dry_run)
        if ret != 0:
            print(f"[ABLT] Experiment '{exp['name']}' failed with exit code {ret}.")
            sys.exit(ret)
    print("[ABLT] All experiments finished successfully.")


if __name__ == "__main__":
    main()
