#!/usr/bin/env python3
"""Regenerate line/bar figures for Matrix-Free Transfer of Quantum Approximate Optimization Parameters Across Circuit Depths."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
BASE_COMMAND = ["topological-qaoa-reproduce","--figures"]


def main() -> None:
    if not BASE_COMMAND:
        raise SystemExit("No separate command is required; see code/README.md.")
    command = list(BASE_COMMAND)
    if command[0] == "python":
        command[0] = sys.executable
    elif shutil.which(command[0]) is None:
        raise SystemExit(
            f"{command[0]!r} is not installed. Run: python -m pip install -e code"
        )
    subprocess.run(command + sys.argv[1:], cwd=REPOSITORY, check=True)


if __name__ == "__main__":
    main()
