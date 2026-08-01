#!/usr/bin/env python3
"""Fail if a theorem check or central claim boundary is absent."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    result = json.loads((repository / "code" / "results" / "results.json").read_text())
    failed = [key for key, value in result["theorem_checks"].items() if not value]
    manuscript = (repository / "main.tex").read_text(encoding="utf-8")
    required = (
        "not a global optimum",
        "no wall-clock speedup",
        "withdraw",
        "feature-signature",
    )
    missing = [token for token in required if token.lower() not in manuscript.lower()]
    if failed or missing:
        raise SystemExit(json.dumps({"failed_theorem_checks": failed, "missing_boundaries": missing}, indent=2))
    print("claim audit: PASS")


if __name__ == "__main__":
    main()
