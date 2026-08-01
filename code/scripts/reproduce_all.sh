#!/usr/bin/env bash
# One-command reproduction of the registered audit.
set -euo pipefail
python scripts/download_data.py --verify-only
python scripts/run.py --data-dir data/source --verify
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/audit_claims.py
python scripts/validate_release.py
echo "==> collision-aware audit reproduced and validated"
