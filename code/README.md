# Reproduction package

`hqml-topological-qaoa` version 2.0.0 reproduces the collision-aware audit in
`../main.tex`. It authenticates eight Stanford Gset files, constructs 144
registered neighborhoods from six topology-unique sources, regenerates the two
grouped evaluations, checks seven mathematical consequences, and emits the
five manuscript tables and five figures.

```bash
python -m pip install .
topological-qaoa-reproduce --verify
python -m pytest -q tests
python scripts/download_data.py --verify-only
python scripts/validate_release.py
```

From a source checkout, regenerate locked evidence with:

```bash
topological-qaoa-reproduce --write --figures
```

The scientific JSON comparison ignores only elapsed-time values, platform
strings, and the absolute data path. All label, split, prediction, operator,
and theorem-check values are compared numerically.

The legacy `experiment.py` remains solely to reconstruct the rejected method
on its original split. The active entry point is `audit.py`. The package does
not claim a globally optimal depth-two schedule, independent observations,
hardware performance, or an elapsed-time advantage.
