#!/usr/bin/env python3
"""Download and authenticate the eight registered Stanford Gset files."""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
from hqml_topological_qaoa.audit import DATASETS, sha256

REPOSITORY = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY / "code" / "data" / "source")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, metadata in DATASETS.items():
        target = args.output_dir / name
        if not target.is_file() or sha256(target) != metadata["sha256"]:
            if args.verify_only:
                raise SystemExit(f"missing or invalid: {target}")
            with urllib.request.urlopen(metadata["url"], timeout=60) as response:
                payload = response.read()
            target.write_bytes(payload)
        actual = sha256(target)
        if actual != metadata["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {name}: {actual}")
        print(f"verified {name} {actual}")


if __name__ == "__main__":
    main()
