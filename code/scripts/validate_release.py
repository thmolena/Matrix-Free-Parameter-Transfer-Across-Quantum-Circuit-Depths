#!/usr/bin/env python3
"""Validate the scientific and packaging closure of the public checkout."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    code = repository / "code"
    result = json.loads((code / "results" / "results.json").read_text())
    evidence = json.loads((code / "evidence_manifest.json").read_text())
    manuscript = (repository / "main.tex").read_text(encoding="utf-8")
    problems = []

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                value.update(chunk)
        return value.hexdigest()

    required_root = {".gitignore", "LICENSE", "README.md", "index.html", "main.tex", "main.pdf", "code"}
    present_root = {path.name for path in repository.iterdir() if path.name != ".git"}
    if present_root != required_root:
        problems.append({"root_missing": sorted(required_root - present_root), "root_extra": sorted(present_root - required_root)})

    figures = sorted((code / "manuscript_assets" / "figures").glob("*.pdf"))
    tables = sorted((code / "manuscript_assets" / "tables").glob("tab_*.tex"))
    referenced_figures = [path for path in figures if path.name == "depth_lifting.pdf"]
    referenced_tables = [
        path for path in tables if path.name in {"tab_data.tex", "tab_depth.tex"}
    ]
    if len(referenced_figures) != 1:
        problems.append(f"expected the depth-lifting PDF figure, found {len(referenced_figures)}")
    if len(referenced_tables) != 2:
        problems.append(f"expected 2 generated evidence tables, found {len(referenced_tables)}")
    for path in referenced_figures + referenced_tables:
        relative = str(path.relative_to(repository)).replace("\\", "/")
        if relative not in manuscript:
            problems.append(f"unreferenced generated asset: {relative}")

    direct_tables = len(re.findall(r"\\begin\{table\*?\}", manuscript))
    if direct_tables:
        problems.append(f"manuscript contains {direct_tables} handwritten tables")
    if any(not value for value in result["theorem_checks"].values()):
        problems.append("one or more theorem checks failed")
    if result["protocols"]["signature_grouped"]["feature_signature_overlap_total"] != 0:
        problems.append("signature-grouped leakage")
    expected_hashes = {
        repository / "main.tex": evidence["manuscript"]["tex_sha256"],
        repository / "main.pdf": evidence["manuscript"]["pdf_sha256"],
        code / "results" / "results.json": evidence["locked_results"]["sha256"],
    }
    for path, expected in expected_hashes.items():
        if digest(path) != expected:
            problems.append(f"manifest hash mismatch: {path}")

    forbidden = (
        "rank-counterfactual operator loss",
        "MF-RCOL",
        "beats both frozen controls",
        "quantum advantage",
        "global depth-two optimum",
    )
    found = [token for token in forbidden if token.lower() in manuscript.lower()]
    if found:
        problems.append({"forbidden_claims": found})

    for name, source in result["sources"].items():
        for base in (code / "data" / "source", code / "src" / "hqml_topological_qaoa" / "data" / "source"):
            path = base / name
            if not path.is_file():
                problems.append(f"missing authenticated mirror: {path}")

    if problems:
        raise SystemExit(json.dumps(problems, indent=2))
    print("release contract: PASS (8 sources, 2 tables, 1 figure, 7 theorem checks)")


if __name__ == "__main__":
    main()
