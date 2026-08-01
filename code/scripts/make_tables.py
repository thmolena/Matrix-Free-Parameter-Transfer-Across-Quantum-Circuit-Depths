#!/usr/bin/env python3
"""Regenerate manuscript tables from the locked audit."""
from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
from hqml_topological_qaoa.audit import write_tables


def main() -> None:
    code = Path(__file__).resolve().parents[1]
    result = json.loads((code / "results" / "results.json").read_text(encoding="utf-8"))
    write_tables(result, code / "manuscript_assets" / "tables")
    print("regenerated five manuscript tables and macros")


if __name__ == "__main__":
    main()
