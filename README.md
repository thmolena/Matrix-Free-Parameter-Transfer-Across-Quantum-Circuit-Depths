# Feature-Conditioned Affine Lifting of QAOA Parameters Across Circuit Depths

This repository contains the manuscript and deterministic CPU reproduction for a feature-conditioned affine map from depth-one QAOA references to depth-two schedules. The evidence uses authenticated Stanford Gset files, excludes duplicate unweighted topologies, and evaluates both source-held-out and collision-free feature-signature folds.

The primary result is protocol-specific. On feature-signature-held-out folds, mean depth-two relative regret is 0.774% for the conditioned lift, 4.050% for an unconditioned affine lift, and 6.632% for interpolation. On source-held-out folds, the unconditioned lift is better than the conditioned lift. The repository preserves both outcomes.

## Reproduction

```bash
python -m venv .venv
.venv/bin/pip install -e 'code[test]'
.venv/bin/topological-qaoa-reproduce --fetch-data --verify
.venv/bin/pytest code/tests
tectonic main.tex
```

The replay authenticates graph bytes, reconstructs the neighborhoods and local reference schedules, runs both split protocols, regenerates every displayed artifact, and verifies all mathematical checks. Wall-clock fields are excluded from deterministic result comparison.

## Scientific scope

The depth-two labels are fixed-budget local references rather than certified global optima. Edge weights are discarded. The study concerns ten-vertex neighborhoods extracted from six topology-unique Gset graphs and does not imply universal cross-family transfer.
