PY ?= python

.PHONY: install test reproduce tables figures audit validate

install:
	$(PY) -m pip install .

test:
	$(PY) -m pytest -q tests

reproduce:
	$(PY) scripts/run.py --data-dir data/source --verify

tables:
	$(PY) scripts/make_tables.py

figures:
	$(PY) scripts/make_figures.py

audit:
	$(PY) scripts/audit_claims.py

validate:
	$(PY) scripts/validate_release.py
