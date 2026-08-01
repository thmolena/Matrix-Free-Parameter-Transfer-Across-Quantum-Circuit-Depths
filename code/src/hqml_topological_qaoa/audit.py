"""Collision-aware, CPU-only audit of QAOA parameter transfer.

The module deliberately separates three questions that the earlier experiment
conflated:

* can a row split be solved by an exact feature duplicate;
* does a predictor generalize to a held-out source or feature signature; and
* does a hierarchical approximation reproduce a specified Gaussian smoother.

All labels are deterministic small-state references.  They are not claimed to
be global QAOA optima.  All Gset bytes are authenticated before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import experiment
from .matrix_free import EntryStats, HNode, build_hierarchy


PACKAGE_DIR = Path(__file__).resolve().parent
CODE_ROOT = Path(__file__).resolve().parents[2]
CHECKOUT = CODE_ROOT if (CODE_ROOT / "pyproject.toml").is_file() else None
RESULTS_ROOT = (CHECKOUT / "results") if CHECKOUT else (PACKAGE_DIR / "results")
DEFAULT_DATA_ROOT = (
    (CHECKOUT / "data" / "source") if CHECKOUT else (PACKAGE_DIR / "data" / "source")
)
SEED = 260731144
ANALYSIS_SOURCES = ("G1", "G11", "G14", "G22", "G32", "G43")
ALL_SOURCES = ("G1", "G6", "G11", "G14", "G18", "G22", "G32", "G43")
RECORDS_PER_SOURCE = 24
PENALTIES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
K_VALUES = (1, 3, 5, 9, 15)
BANDWIDTHS = (0.03, 0.05, 0.08, 0.12, 0.20)
ALPHAS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0)

DATASETS = {
    "G1": {
        "sha256": "73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665",
        "url": "https://web.stanford.edu/~yyye/yyye/Gset/G1",
    },
    "G6": {
        "sha256": "e3dd8b29205280f1ad5806849321cdff6f11324c75cb95647d5783f3e47a1103",
        "url": "https://web.stanford.edu/~yyye/yyye/Gset/G6",
    },
    "G11": {
        "sha256": "c2a760d2926db4fefd23b25c098dcd6311f711b355dbd1cc689fa25660c73174",
        "url": "https://web.stanford.edu/~yyye/yyye/Gset/G11",
    },
    "G14": {
        "sha256": "dc769b978a40d458f693d5bd2cf8b8cceabd430b8e976204746696179c3d5945",
        "url": "https://web.stanford.edu/~yyye/yyye/Gset/G14",
    },
    "G18": {
        "sha256": "6185af31fae0f7d59e232dad91178753574b7caa111dd6e6bf52f11c80e928d8",
        "url": "https://web.stanford.edu/~yyye/yyye/Gset/G18",
    },
    "G22": {
        "sha256": "9baeee06eb147b1c9ca42b43be86592d4e6fc60784a85af9be5b63d1362ef28e",
        "url": "https://web.stanford.edu/~yyye/yyye/Gset/G22",
    },
    "G32": {
        "sha256": "9760fce6b601832db1b1173c1a09c8556d64bfb27ba603ef52ca24ee6c8527cc",
        "url": "https://web.stanford.edu/~yyye/yyye/Gset/G32",
    },
    "G43": {
        "sha256": "9af5445b4b066cbf1eabe218d4e0d907cb6f211651cae557c761ec344dc37be8",
        "url": "https://web.stanford.edu/~yyye/yyye/Gset/G43",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_root(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.environ.get("HQML_GSET_CACHE")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DATA_ROOT


def topology_sha256(n: int, weighted_edges: list[tuple[int, int, float]]) -> str:
    lines = [f"{n}\n"]
    lines.extend(f"{min(u, v)} {max(u, v)}\n" for u, v, _ in sorted(weighted_edges))
    return hashlib.sha256("".join(lines).encode("ascii")).hexdigest()


def authenticate_sources(data_root: Path) -> tuple[dict[str, dict], dict[str, tuple]]:
    inventory: dict[str, dict] = {}
    parsed: dict[str, tuple] = {}
    for name in ALL_SOURCES:
        path = data_root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; run code/scripts/download_data.py or pass --data-dir"
            )
        actual = sha256(path)
        expected = DATASETS[name]["sha256"]
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch: {actual} != {expected}")
        n, edges = experiment.parse_gset(path)
        topology = topology_sha256(n, edges)
        parsed[name] = (n, edges)
        inventory[name] = {
            "sha256": actual,
            "topology_sha256": topology,
            "vertices": n,
            "edges": len(edges),
            "url": DATASETS[name]["url"],
            "role": "analysis" if name in ANALYSIS_SOURCES else "excluded duplicate topology",
        }
    topology_groups: dict[str, list[str]] = {}
    for name, item in inventory.items():
        topology_groups.setdefault(item["topology_sha256"], []).append(name)
    duplicates = sorted(tuple(names) for names in topology_groups.values() if len(names) > 1)
    if duplicates != [("G1", "G6"), ("G14", "G18")]:
        raise AssertionError(f"unexpected topology duplicate groups: {duplicates}")
    return inventory, parsed


def local_records(
    name: str,
    parsed_source: tuple[int, list[tuple[int, int, float]]],
    count: int,
) -> list[dict]:
    n_source, weighted_edges = parsed_source
    adjacency = [[] for _ in range(n_source)]
    for u, v, _ in weighted_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbors in adjacency:
        neighbors.sort()

    records = []
    for sample in range(count):
        start = (sample * 37) % n_source
        vertices = experiment.bfs_vertices(adjacency, start, size=10)
        local_index = {vertex: index for index, vertex in enumerate(vertices)}
        edges = [
            (local_index[u], local_index[v])
            for u in vertices
            for v in adjacency[u]
            if u < v and v in local_index
        ]
        edges.sort()
        diagonal = experiment.cost_diagonal(edges, len(vertices))
        p1, value1 = experiment.p1_grid_reference(edges, len(vertices))
        p2, value2, queries = experiment.p2_refined_reference(diagonal, p1, len(vertices))
        records.append(
            {
                "source": name,
                "start": start,
                "features": experiment.graph_features(edges, len(vertices)),
                "target": experiment.encode_schedule(p1, p2),
                "diagonal": diagonal,
                "edges_local": edges,
                "raw_signature": tuple(edges),
                "p1_value": value1,
                "p2_value": value2,
                "p2_queries": queries,
            }
        )
    return records


def feature_signature(row: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(row, dtype=float), 12))


def balanced_signature_folds(features: np.ndarray, fold_count: int = 6) -> list[np.ndarray]:
    groups: dict[tuple[float, ...], list[int]] = {}
    for index, row in enumerate(features):
        groups.setdefault(feature_signature(row), []).append(index)
    bins: list[list[int]] = [[] for _ in range(fold_count)]
    sizes = [0] * fold_count
    for indices in sorted(groups.values(), key=lambda item: (-len(item), item[0])):
        destination = min(range(fold_count), key=lambda fold: (sizes[fold], fold))
        bins[destination].extend(indices)
        sizes[destination] += len(indices)
    return [np.asarray(sorted(indices), dtype=int) for indices in bins]


def standardize(features: np.ndarray, train: np.ndarray) -> np.ndarray:
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    active = scale >= 1e-12
    if not np.any(active):
        raise ValueError("all features are constant on the training split")
    return (features[:, active] - mean[active]) / scale[active]


def scalar_key(features: np.ndarray, train: np.ndarray) -> np.ndarray:
    training = features[train]
    covariance = training.T @ training / max(len(training), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    raw = features @ direction
    lower = float(raw[train].min())
    span = max(float(np.ptp(raw[train])), 1e-12)
    return (raw - lower) / span


class GaussianEntryOracle:
    """Nonnegative Gaussian kernel queried by rectangular blocks."""

    def __init__(self, key: np.ndarray, bandwidth: float):
        self.key = np.asarray(key, dtype=float).reshape(-1)
        self.bandwidth = float(bandwidth)
        self.stats = EntryStats()

    @property
    def n(self) -> int:
        return len(self.key)

    def block(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        entries = int(len(rows) * len(cols))
        self.stats.calls += 1
        self.stats.queried_entries += entries
        self.stats.peak_block_entries = max(self.stats.peak_block_entries, entries)
        delta = self.key[rows, None] - self.key[None, cols]
        return np.exp(-0.5 * (delta / self.bandwidth) ** 2)


def exact_kernel(key: np.ndarray, bandwidth: float) -> np.ndarray:
    delta = key[:, None] - key[None, :]
    return np.exp(-0.5 * (delta / bandwidth) ** 2)


def normalized_prediction(
    matrix: np.ndarray,
    residual: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    payload = residual * train_mask[:, None]
    numerator = matrix @ payload
    denominator = matrix @ train_mask.astype(float)
    fallback = residual[train_mask].mean(axis=0)
    return np.divide(
        numerator,
        denominator[:, None],
        out=np.tile(fallback, (len(matrix), 1)),
        where=np.abs(denominator[:, None]) > 1e-12,
    )


def exact_prior(
    key: np.ndarray,
    residual: np.ndarray,
    train_mask: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    return normalized_prediction(exact_kernel(key, bandwidth), residual, train_mask)


def hierarchical_prior(
    key: np.ndarray,
    residual: np.ndarray,
    train_mask: np.ndarray,
    bandwidth: float,
    *,
    rank: int = 8,
    leaf_size: int = 16,
) -> tuple[np.ndarray, dict]:
    order = np.argsort(key, kind="mergesort")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    oracle = GaussianEntryOracle(key[order], bandwidth)
    tree = build_hierarchy(
        oracle, np.arange(len(order)), leaf_size=leaf_size, rank=rank
    )
    mask = train_mask[order]
    values = residual[order]
    payload = np.column_stack((values * mask[:, None], mask.astype(float)))
    output, work = tree.apply(payload)
    fallback = residual[train_mask].mean(axis=0)
    prediction = np.divide(
        output[:, :-1],
        output[:, -1:],
        out=np.tile(fallback, (len(order), 1)),
        where=np.abs(output[:, -1:]) > 1e-12,
    )
    return prediction[inverse], {
        "storage": tree.storage,
        "work": work,
        "queried_entries": oracle.stats.queried_entries,
        "peak_block_entries": oracle.stats.peak_block_entries,
    }


def weighted_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(x_eval[:, None, :] - x_train[None, :, :], axis=2)
    neighbors = np.argsort(distances, axis=1, kind="mergesort")[:, :k]
    predictions = []
    for row, selected in enumerate(neighbors):
        distance = distances[row, selected]
        if distance[0] < 1e-12:
            # An exact representation collision can contain more than k
            # donors.  Average the entire zero-distance equivalence class so
            # the result is independent of the sorting algorithm's tie order.
            exact = np.flatnonzero(distances[row] < 1e-12)
            predictions.append(np.mean(y_train[exact], axis=0))
            continue
        else:
            weights = 1.0 / np.maximum(distance, 1e-12)
        weights /= weights.sum()
        predictions.append(np.sum(y_train[selected] * weights[:, None], axis=0))
    return np.asarray(predictions), distances.min(axis=1)


def prediction_metrics(
    prediction: np.ndarray,
    indices: np.ndarray,
    records: list[dict],
) -> dict:
    target = np.asarray([records[index]["target"] for index in indices])
    p1_regrets, p2_regrets = [], []
    for row, index in enumerate(indices):
        p1, p2 = experiment.decode_schedule(prediction[row])
        record = records[index]
        value1 = experiment.p1_closed_form(record["edges_local"], p1[0], p1[1], 10)
        value2 = experiment.schedule_energy(record["diagonal"], p2[:2], p2[2:], 10)
        p1_regrets.append(max(0.0, (record["p1_value"] - value1) / record["p1_value"]))
        p2_regrets.append(max(0.0, (record["p2_value"] - value2) / record["p2_value"]))
    return {
        "schedule_mse": float(np.mean((prediction - target) ** 2)),
        "p1_regret": float(np.mean(p1_regrets)),
        "p2_regret": float(np.mean(p2_regrets)),
        "p1_median_regret": float(np.median(p1_regrets)),
        "p2_median_regret": float(np.median(p2_regrets)),
    }


def p2_metrics(prediction: np.ndarray, indices: np.ndarray, records: list[dict]) -> dict:
    target = np.asarray([records[index]["target"][2:] for index in indices])
    regrets = []
    for row, index in enumerate(indices):
        record = records[index]
        _, p2 = experiment.decode_schedule(np.r_[record["target"][:2], prediction[row]])
        value = experiment.schedule_energy(record["diagonal"], p2[:2], p2[2:], 10)
        regrets.append(max(0.0, (record["p2_value"] - value) / record["p2_value"]))
    return {
        "p2_schedule_mse": float(np.mean((prediction - target) ** 2)),
        "p2_regret": float(np.mean(regrets)),
        "p2_median_regret": float(np.median(regrets)),
    }


def tune_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    development: np.ndarray,
) -> float:
    losses = {}
    for penalty in PENALTIES:
        prediction = experiment.ridge_fit_predict(
            features, targets, train, development, penalty
        )
        losses[penalty] = float(np.mean((prediction - targets[development]) ** 2))
    return min(PENALTIES, key=lambda value: (losses[value], value))


def tune_correction(
    baseline: np.ndarray,
    truth: np.ndarray,
    development: np.ndarray,
    priors: dict[float, np.ndarray],
) -> tuple[float, float]:
    candidates = []
    for bandwidth, prior in priors.items():
        for alpha in ALPHAS:
            prediction = np.clip(
                baseline[development] + alpha * prior[development], 0.0, 1.0
            )
            loss = float(np.mean((prediction - truth[development]) ** 2))
            candidates.append((loss, bandwidth, alpha))
    _, bandwidth, alpha = min(candidates)
    return bandwidth, alpha


def _fold(
    records: list[dict],
    features: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    development: np.ndarray,
    test: np.ndarray,
) -> tuple[dict, dict]:
    standardized = standardize(features, train)
    ridge_penalty = tune_ridge(standardized, targets, train, development)
    all_indices = np.arange(len(records))
    ridge = experiment.ridge_fit_predict(
        standardized, targets, train, all_indices, ridge_penalty
    )
    residual = targets - ridge
    train_mask = np.zeros(len(records), dtype=bool)
    train_mask[train] = True
    key = scalar_key(standardized, train)

    k_losses = {}
    for k in K_VALUES:
        candidate, _ = weighted_knn(
            standardized[train], targets[train], standardized[development], k
        )
        k_losses[k] = float(np.mean((candidate - targets[development]) ** 2))
    selected_k = min(K_VALUES, key=lambda value: (k_losses[value], value))
    knn, nearest = weighted_knn(
        standardized[train], targets[train], standardized[test], selected_k
    )

    dense_priors = {
        bandwidth: exact_prior(key, residual, train_mask, bandwidth)
        for bandwidth in BANDWIDTHS
    }
    dense_bandwidth, dense_alpha = tune_correction(
        ridge, targets, development, dense_priors
    )
    dense = np.clip(
        ridge[test] + dense_alpha * dense_priors[dense_bandwidth][test], 0.0, 1.0
    )

    hierarchy_priors, hierarchy_resources = {}, {}
    for bandwidth in BANDWIDTHS:
        prior, resources = hierarchical_prior(
            key, residual, train_mask, bandwidth, rank=8, leaf_size=16
        )
        hierarchy_priors[bandwidth] = prior
        hierarchy_resources[bandwidth] = resources
    hierarchy_bandwidth, hierarchy_alpha = tune_correction(
        ridge, targets, development, hierarchy_priors
    )
    hierarchy = np.clip(
        ridge[test]
        + hierarchy_alpha * hierarchy_priors[hierarchy_bandwidth][test],
        0.0,
        1.0,
    )

    constant = np.tile(targets[train].mean(axis=0), (len(test), 1))
    structure_predictions = {
        "constant": constant,
        "ridge": ridge[test],
        "weighted_knn": knn,
        "dense_residual": dense,
        "hierarchical_residual": hierarchy,
    }

    p1 = targets[:, :2]
    interp = np.column_stack(
        (p1[test, 0] / 2, p1[test, 0] / 2, p1[test, 1] / 2, p1[test, 1] / 2)
    )
    p1_penalty = tune_ridge(p1, targets[:, 2:], train, development)
    p1_affine = experiment.ridge_fit_predict(
        p1, targets[:, 2:], train, test, p1_penalty
    )
    p1_and_features = np.column_stack((p1, standardized))
    p1_feature_penalty = tune_ridge(
        p1_and_features, targets[:, 2:], train, development
    )
    p1_feature_affine = experiment.ridge_fit_predict(
        p1_and_features, targets[:, 2:], train, test, p1_feature_penalty
    )
    depth_predictions = {
        "interp": interp,
        "p1_affine": p1_affine,
        "p1_feature_affine": p1_feature_affine,
    }

    train_signatures = {feature_signature(features[index]) for index in train}
    test_signatures = {feature_signature(features[index]) for index in test}
    report = {
        "train_size": len(train),
        "development_size": len(development),
        "test_size": len(test),
        "feature_signature_overlap": len(train_signatures & test_signatures),
        "zero_distance_test_records": int(np.sum(nearest < 1e-12)),
        "selected_ridge_penalty": ridge_penalty,
        "selected_knn_k": selected_k,
        "selected_dense_bandwidth": dense_bandwidth,
        "selected_dense_alpha": dense_alpha,
        "selected_hierarchy_bandwidth": hierarchy_bandwidth,
        "selected_hierarchy_alpha": hierarchy_alpha,
        "hierarchy_storage_fraction": hierarchy_resources[hierarchy_bandwidth]["storage"]
        / len(records) ** 2,
        "methods": {
            name: prediction_metrics(prediction, test, records)
            for name, prediction in structure_predictions.items()
        },
        "depth_methods": {
            name: p2_metrics(prediction, test, records)
            for name, prediction in depth_predictions.items()
        },
    }
    artifacts = {
        "indices": test,
        "structure_predictions": structure_predictions,
        "depth_predictions": depth_predictions,
    }
    return report, artifacts


def evaluate_protocol(
    name: str,
    records: list[dict],
    fold_triplets: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[dict, dict]:
    features = np.asarray([record["features"] for record in records])
    targets = np.asarray([record["target"] for record in records])
    fold_reports, all_indices = [], []
    structure = {name: [] for name in (
        "constant", "ridge", "weighted_knn", "dense_residual", "hierarchical_residual"
    )}
    depth = {name: [] for name in ("interp", "p1_affine", "p1_feature_affine")}
    for fold_number, (train, development, test) in enumerate(fold_triplets):
        fold_report, artifacts = _fold(
            records, features, targets, train, development, test
        )
        fold_report["fold"] = fold_number
        fold_reports.append(fold_report)
        all_indices.extend(test.tolist())
        for method, prediction in artifacts["structure_predictions"].items():
            structure[method].append(prediction)
        for method, prediction in artifacts["depth_predictions"].items():
            depth[method].append(prediction)

    indices = np.asarray(all_indices, dtype=int)
    aggregate = {
        method: prediction_metrics(np.vstack(predictions), indices, records)
        for method, predictions in structure.items()
    }
    aggregate_depth = {
        method: p2_metrics(np.vstack(predictions), indices, records)
        for method, predictions in depth.items()
    }
    report = {
        "name": name,
        "folds": fold_reports,
        "aggregate": aggregate,
        "aggregate_depth": aggregate_depth,
        "total_test_records": len(indices),
        "feature_signature_overlap_total": sum(
            fold["feature_signature_overlap"] for fold in fold_reports
        ),
        "zero_distance_test_records_total": sum(
            fold["zero_distance_test_records"] for fold in fold_reports
        ),
    }
    artifacts = {
        "indices": indices,
        "depth_predictions": {
            method: np.vstack(predictions) for method, predictions in depth.items()
        },
    }
    return report, artifacts


def source_loso_triplets(records: list[dict]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    sources = np.asarray([record["source"] for record in records])
    triplets = []
    for fold, test_source in enumerate(ANALYSIS_SOURCES):
        development_source = ANALYSIS_SOURCES[(fold + 1) % len(ANALYSIS_SOURCES)]
        test = np.flatnonzero(sources == test_source)
        development = np.flatnonzero(sources == development_source)
        train = np.flatnonzero(
            (sources != test_source) & (sources != development_source)
        )
        triplets.append((train, development, test))
    return triplets


def signature_triplets(features: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    bins = balanced_signature_folds(features, fold_count=6)
    universe = np.arange(len(features))
    triplets = []
    for fold, test in enumerate(bins):
        development = bins[(fold + 1) % len(bins)]
        train = np.setdiff1d(universe, np.r_[test, development])
        triplets.append((train, development, test))
    return triplets


def collision_curve(features: np.ndarray) -> dict:
    counts = Counter(feature_signature(row) for row in features)
    sizes = np.asarray(list(counts.values()), dtype=int)
    n = int(sizes.sum())
    probabilities = np.linspace(0.10, 0.80, 8)
    rng = np.random.default_rng(SEED + 17)
    trials = 20_000
    rows = []
    for tau in probabilities:
        theory = 1.0 - float(
            np.sum(sizes * np.power(tau, sizes - 1)) / n
        )
        sampled_sizes = rng.choice(sizes, size=trials, p=sizes / n)
        event_probability = 1.0 - np.power(tau, sampled_sizes - 1)
        empirical = float(np.mean(rng.random(trials) < event_probability))
        standard_error = math.sqrt(max(empirical * (1.0 - empirical), 0.0) / trials)
        rows.append(
            {
                "test_probability": float(tau),
                "theory": theory,
                "simulation": empirical,
                "simulation_standard_error": standard_error,
            }
        )
    return {
        "records": n,
        "feature_signature_classes": len(sizes),
        "largest_class": int(sizes.max()),
        "class_sizes": sorted((int(value) for value in sizes), reverse=True),
        "trials_per_probability": trials,
        "curve": rows,
        "max_simulation_error": max(abs(row["theory"] - row["simulation"]) for row in rows),
    }


def legacy_row_split(records: list[dict]) -> dict:
    features = np.asarray([record["features"] for record in records])
    targets = np.asarray([record["target"] for record in records])
    indices = np.arange(len(records))
    test = indices[indices % 4 == 0]
    train = indices[indices % 4 != 0]
    development = train[::5]
    fit = np.setdiff1d(train, development)
    standardized = standardize(features, fit)
    k_losses = {}
    for k in K_VALUES:
        prediction, _ = weighted_knn(
            standardized[fit], targets[fit], standardized[development], k
        )
        k_losses[k] = float(np.mean((prediction - targets[development]) ** 2))
    selected_k = min(K_VALUES, key=lambda value: (k_losses[value], value))
    knn, nearest = weighted_knn(
        standardized[train], targets[train], standardized[test], selected_k
    )
    train_raw = {records[index]["raw_signature"] for index in train}
    old = experiment.run_experiment()
    return {
        "train_records": len(train),
        "development_records": len(development),
        "test_records": len(test),
        "unique_feature_signatures": len(
            {feature_signature(row) for row in features}
        ),
        "unique_target_schedules": len(
            {feature_signature(row) for row in targets}
        ),
        "feature_collision_test_records": int(np.sum(nearest < 1e-12)),
        "raw_neighborhood_duplicate_test_records": sum(
            records[index]["raw_signature"] in train_raw for index in test
        ),
        "selected_knn_k": selected_k,
        "methods": {
            "constant": {
                "schedule_mse": old["constant_schedule_mse"],
                "p1_regret": old["constant_regret_p1"],
                "p2_regret": old["constant_regret_p2"],
            },
            "ridge": {
                "schedule_mse": old["ridge_schedule_mse"],
                "p1_regret": old["ridge_regret_p1"],
                "p2_regret": old["ridge_regret_p2"],
            },
            "rejected_rcol": {
                "schedule_mse": old["rcol_schedule_mse"],
                "p1_regret": old["rcol_regret_p1"],
                "p2_regret": old["rcol_regret_p2"],
            },
            "weighted_knn": prediction_metrics(knn, test, records),
        },
    }


def _tree_matrix(tree: HNode) -> tuple[np.ndarray, int]:
    return tree.apply(np.eye(tree.n))


def _median_seconds(function: Callable[[], object], loops: int, repeats: int = 7) -> float:
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(loops):
            function()
        samples.append((time.perf_counter() - started) / loops)
    return float(statistics.median(samples))


def operator_audit(records: list[dict]) -> dict:
    features = np.asarray([record["features"] for record in records])
    targets = np.asarray([record["target"] for record in records])
    sources = np.asarray([record["source"] for record in records])
    train = np.flatnonzero((sources != "G1") & (sources != "G11"))
    standardized = standardize(features, train)
    key = scalar_key(standardized, train)
    train_mask = np.zeros(len(records), dtype=bool)
    train_mask[train] = True
    residual = targets - targets[train].mean(axis=0)
    bandwidth = 0.08
    dense = exact_kernel(key, bandwidth)
    exact_prediction = normalized_prediction(dense, residual, train_mask)
    rank_rows = []
    for rank in (1, 2, 4, 8, 12):
        order = np.argsort(key, kind="mergesort")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        oracle = GaussianEntryOracle(key[order], bandwidth)
        tree = build_hierarchy(
            oracle, np.arange(len(order)), leaf_size=16, rank=rank
        )
        approximate_sorted, work_identity = _tree_matrix(tree)
        approximate = approximate_sorted[np.ix_(inverse, inverse)]
        approximate_prediction = normalized_prediction(
            approximate, residual, train_mask
        )
        restricted_error = np.abs(approximate - dense)[:, train_mask]
        epsilon = restricted_error.sum(axis=1)
        denominator = dense[:, train_mask].sum(axis=1)
        residual_norm = np.linalg.norm(residual[train_mask], axis=1)
        residual_bound = float(residual_norm.max())
        certified = epsilon < denominator
        bounds = np.full(len(records), np.nan)
        bounds[certified] = (
            2.0
            * residual_bound
            * epsilon[certified]
            / (denominator[certified] - epsilon[certified])
        )
        actual = np.linalg.norm(approximate_prediction - exact_prediction, axis=1)
        violations = int(np.sum(actual[certified] > bounds[certified] + 1e-10))
        _, apply_work = tree.apply(np.ones((len(records), 7)))
        levels = math.ceil(math.log2(len(records) / 16))
        storage_bound = len(records) * (16 + rank * levels)
        work_bound = 7 * len(records) * (16 + 2 * rank * levels)
        rank_rows.append(
            {
                "rank": rank,
                "relative_frobenius_error": float(
                    np.linalg.norm(approximate - dense) / np.linalg.norm(dense)
                ),
                "storage_scalars": tree.storage,
                "storage_fraction": tree.storage / len(records) ** 2,
                "storage_upper_bound": storage_bound,
                "apply_work": apply_work,
                "apply_work_upper_bound": work_bound,
                "entry_query_fraction": oracle.stats.queried_entries / len(records) ** 2,
                "certified_rows": int(np.sum(certified)),
                "uncertified_rows": int(np.sum(~certified)),
                "max_actual_prediction_error": float(actual.max()),
                "max_finite_prediction_bound": float(np.nanmax(bounds)),
                "prediction_bound_violations": violations,
                "identity_apply_work": work_identity,
            }
        )

    scaling = []
    for n in (36, 72, 108, 144):
        current = standardized[:n]
        train_n = np.arange(n)
        key_n = scalar_key(current, train_n)
        order = np.argsort(key_n, kind="mergesort")
        key_sorted = key_n[order]
        rhs = np.linspace(-1.0, 1.0, n * 7).reshape(n, 7)
        dense_n = exact_kernel(key_sorted, bandwidth)
        oracle = GaussianEntryOracle(key_sorted, bandwidth)
        tree = build_hierarchy(oracle, np.arange(n), leaf_size=16, rank=8)
        _, work = tree.apply(rhs)
        dense_apply = _median_seconds(lambda: dense_n @ rhs, loops=300)
        hierarchy_apply = _median_seconds(lambda: tree.apply(rhs), loops=100)
        dense_build = _median_seconds(
            lambda: exact_kernel(key_sorted, bandwidth), loops=100
        )
        hierarchy_build = _median_seconds(
            lambda: build_hierarchy(
                GaussianEntryOracle(key_sorted, bandwidth),
                np.arange(n),
                leaf_size=16,
                rank=8,
            ),
            loops=10,
        )
        levels = math.ceil(math.log2(n / 16)) if n > 16 else 0
        scaling.append(
            {
                "records": n,
                "dense_storage_scalars": n * n,
                "hierarchy_storage_scalars": tree.storage,
                "storage_upper_bound": n * (16 + 8 * levels),
                "dense_apply_work": n * n * 7,
                "hierarchy_apply_work": work,
                "apply_work_upper_bound": 7 * n * (16 + 16 * levels),
                "dense_apply_seconds": dense_apply,
                "hierarchy_apply_seconds": hierarchy_apply,
                "dense_build_seconds": dense_build,
                "hierarchy_build_seconds": hierarchy_build,
            }
        )
    return {
        "kernel": "one-dimensional Gaussian",
        "bandwidth": bandwidth,
        "leaf_size": 16,
        "right_hand_sides": 7,
        "rank_audit": rank_rows,
        "scaling": scaling,
        "runtime_environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def qaoa_stability_audit(
    records: list[dict], indices: np.ndarray, predictions: np.ndarray
) -> dict:
    points = []
    violations = 0
    for row, index in enumerate(indices):
        record = records[index]
        target = record["target"]
        _, p2_reference = experiment.decode_schedule(target)
        _, p2_prediction = experiment.decode_schedule(
            np.r_[target[:2], predictions[row]]
        )
        gamma_error = float(np.sum(np.abs(p2_prediction[:2] - p2_reference[:2])))
        beta_error = float(np.sum(np.abs(p2_prediction[2:] - p2_reference[2:])))
        edges = len(record["edges_local"])
        bound = (
            2.0 * edges**2 * gamma_error + 2.0 * 10 * edges * beta_error
        ) / record["p2_value"]
        achieved = experiment.schedule_energy(
            record["diagonal"], p2_prediction[:2], p2_prediction[2:], 10
        )
        gap = abs(record["p2_value"] - achieved) / record["p2_value"]
        violations += int(gap > bound + 1e-10)
        points.append(
            {
                "source": record["source"],
                "gamma_l1_error": gamma_error,
                "beta_l1_error": beta_error,
                "absolute_relative_energy_gap": gap,
                "lipschitz_bound": bound,
            }
        )
    return {
        "records": len(points),
        "violations": violations,
        "max_absolute_relative_energy_gap": max(
            point["absolute_relative_energy_gap"] for point in points
        ),
        "median_lipschitz_bound": float(
            np.median([point["lipschitz_bound"] for point in points])
        ),
        "points": points,
    }


def rounded(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 12)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    return value


def run_audit(data_root: Path) -> dict:
    started = time.perf_counter()
    inventory, parsed = authenticate_sources(data_root)
    records = [
        record
        for name in ANALYSIS_SOURCES
        for record in local_records(name, parsed[name], RECORDS_PER_SOURCE)
    ]
    legacy_records = local_records("G1", parsed["G1"], 256)
    features = np.asarray([record["features"] for record in records])

    source_report, _ = evaluate_protocol(
        "source_leave-one-out", records, source_loso_triplets(records)
    )
    signature_report, signature_artifacts = evaluate_protocol(
        "feature_signature_grouped",
        records,
        signature_triplets(features),
    )
    stability = qaoa_stability_audit(
        records,
        signature_artifacts["indices"],
        signature_artifacts["depth_predictions"]["p1_feature_affine"],
    )
    collision = collision_curve(
        np.asarray([record["features"] for record in legacy_records])
    )
    legacy = legacy_row_split(legacy_records)
    operator = operator_audit(records)

    theorem_checks = {
        "collision_formula_simulation_error_below_0_015": collision[
            "max_simulation_error"
        ]
        < 0.015,
        "signature_group_split_has_zero_overlap": signature_report[
            "feature_signature_overlap_total"
        ]
        == 0,
        "signature_group_split_has_zero_exact_neighbor": signature_report[
            "zero_distance_test_records_total"
        ]
        == 0,
        "kernel_prediction_bound_has_zero_violations": all(
            row["prediction_bound_violations"] == 0
            for row in operator["rank_audit"]
        ),
        "hierarchy_storage_bound_holds": all(
            row["storage_scalars"] <= row["storage_upper_bound"]
            for row in operator["rank_audit"]
        )
        and all(
            row["hierarchy_storage_scalars"] <= row["storage_upper_bound"]
            for row in operator["scaling"]
        ),
        "hierarchy_work_bound_holds": all(
            row["apply_work"] <= row["apply_work_upper_bound"]
            for row in operator["rank_audit"]
        )
        and all(
            row["hierarchy_apply_work"] <= row["apply_work_upper_bound"]
            for row in operator["scaling"]
        ),
        "qaoa_lipschitz_bound_has_zero_violations": stability["violations"] == 0,
    }
    if not all(theorem_checks.values()):
        raise AssertionError(f"theorem check failed: {theorem_checks}")

    return rounded(
        {
            "schema_version": 4,
            "package": "hqml_topological_qaoa",
            "study": "collision-aware QAOA depth-transfer audit",
            "seed": SEED,
            "data_root": str(data_root),
            "sources": inventory,
            "analysis_sources": ANALYSIS_SOURCES,
            "excluded_topology_duplicates": {"G6": "G1", "G18": "G14"},
            "records_per_source": RECORDS_PER_SOURCE,
            "records": len(records),
            "unique_feature_signatures": len(
                {feature_signature(row) for row in features}
            ),
            "label_contract": {
                "vertices_per_neighborhood": 10,
                "weights_used": False,
                "p1_grid_queries": 13 * 9,
                "p2_local_refinement_queries": 25,
                "p2_is_global_optimum": False,
            },
            "legacy_row_split": legacy,
            "collision": collision,
            "protocols": {
                "source_leave_one_out": source_report,
                "signature_grouped": signature_report,
            },
            "operator_audit": operator,
            "qaoa_stability": stability,
            "theorem_checks": theorem_checks,
            "cpu_seconds": time.perf_counter() - started,
        }
    )


def _without_timing(value):
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if not key.endswith("_seconds")
            and key not in {"runtime_environment", "data_root"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _assert_close(actual, expected, path: str = "root") -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{path} keys differ")
        for key in actual:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{path} lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(actual, float) or isinstance(expected, float):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-9):
            raise AssertionError(f"{path}: {actual} != {expected}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def verify(result: dict) -> None:
    expected_path = RESULTS_ROOT / "results.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    _assert_close(_without_timing(result), _without_timing(expected))


def tex_macros(result: dict) -> str:
    legacy = result["legacy_row_split"]
    source = result["protocols"]["source_leave_one_out"]
    grouped = result["protocols"]["signature_grouped"]
    macros = {
        "AuditRecords": result["records"],
        "AuditSources": len(result["analysis_sources"]),
        "FeatureClasses": result["unique_feature_signatures"],
        "LegacyFeatureCollisions": legacy["feature_collision_test_records"],
        "LegacyTestRecords": legacy["test_records"],
        "LegacyRawDuplicates": legacy["raw_neighborhood_duplicate_test_records"],
        "LegacyRCOLMSE": f'{legacy["methods"]["rejected_rcol"]["schedule_mse"]:.3e}',
        "LegacyKNNMSE": f'{legacy["methods"]["weighted_knn"]["schedule_mse"]:.3e}',
        "SourceKNNMSE": f'{source["aggregate"]["weighted_knn"]["schedule_mse"]:.3e}',
        "SourceHRegret": f'{100*source["aggregate"]["hierarchical_residual"]["p2_regret"]:.3f}',
        "SourceKNNRegret": f'{100*source["aggregate"]["weighted_knn"]["p2_regret"]:.3f}',
        "GroupedKNNMSE": f'{grouped["aggregate"]["weighted_knn"]["schedule_mse"]:.3e}',
        "GroupedHMSE": f'{grouped["aggregate"]["hierarchical_residual"]["schedule_mse"]:.3e}',
        "GroupedKNNRegret": f'{100*grouped["aggregate"]["weighted_knn"]["p2_regret"]:.3f}',
        "GroupedHRegret": f'{100*grouped["aggregate"]["hierarchical_residual"]["p2_regret"]:.3f}',
        "GroupedAffineRegret": f'{100*grouped["aggregate_depth"]["p1_feature_affine"]["p2_regret"]:.3f}',
    }
    return "\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items()
    ) + "\n"


def table_data(result: dict) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{\label{tab:data}Authenticated Gset sources. Topology identifiers are SHA-256 prefixes after discarding weights; sources with the same identifier have identical unweighted topology. Only one member of each duplicate pair enters the audit.}",
        r"\centering\small",
        r"\begin{tabular}{lrrrrll}",
        r"\toprule",
        r"Source & Vertices & Edges & Records & File SHA-256 & Topology SHA-256 & Role\\",
        r"\midrule",
    ]
    for name in ALL_SOURCES:
        item = result["sources"][name]
        lines.append(
            f'{name} & {item["vertices"]} & {item["edges"]} & '
            f'{RECORDS_PER_SOURCE if name in ANALYSIS_SOURCES else 0} & '
            rf'\texttt{{{item["sha256"][:10]}}} & '
            rf'\texttt{{{item["topology_sha256"][:10]}}} & {item["role"]}' + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def table_protocols(result: dict) -> str:
    legacy = result["legacy_row_split"]
    source = result["protocols"]["source_leave_one_out"]
    grouped = result["protocols"]["signature_grouped"]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{\label{tab:protocols}Evaluation protocols. Fold sizes are ranges across the six rotating folds. Overlap counts feature signatures shared by a training and test partition.}",
            r"\centering\scriptsize",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Protocol & Test/fold & Sig. overlap & Zero-dist.\\",
            r"\midrule",
            f'Legacy row & {legacy["test_records"]} & -- & {legacy["feature_collision_test_records"]}' + r"\\",
            f'Source LOSO & 24 & {source["feature_signature_overlap_total"]} & {source["zero_distance_test_records_total"]}' + r"\\",
            f'Signature groups & 15--46 & {grouped["feature_signature_overlap_total"]} & {grouped["zero_distance_test_records_total"]}' + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def table_results(result: dict) -> str:
    methods = (
        ("constant", "Constant"),
        ("ridge", "Ridge"),
        ("weighted_knn", "Weighted nearest donor"),
        ("dense_residual", "Dense residual smoother"),
        ("hierarchical_residual", "Hierarchical residual smoother"),
    )
    lines = [
        r"\begin{table*}[t]",
        r"\caption{\label{tab:results}Structure-to-schedule evaluation. Values aggregate each record once across rotating folds. MSE is over six normalized schedule coordinates; $R_1$ and $R_2$ are mean reference-relative regrets. No uncertainty claim is made from dependent neighborhoods.}",
        r"\centering\small",
        r"\begin{tabular}{lrrr@{\qquad}rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Source leave-one-out} & \multicolumn{3}{c}{Signature grouped}\\",
        r"Method & MSE & $R_1$ (\%) & $R_2$ (\%) & MSE & $R_1$ (\%) & $R_2$ (\%)\\",
        r"\midrule",
    ]
    for key, label in methods:
        left = result["protocols"]["source_leave_one_out"]["aggregate"][key]
        right = result["protocols"]["signature_grouped"]["aggregate"][key]
        lines.append(
            f'{label} & {left["schedule_mse"]:.3e} & {100*left["p1_regret"]:.3f} & {100*left["p2_regret"]:.3f} & '
            f'{right["schedule_mse"]:.3e} & {100*right["p1_regret"]:.3f} & {100*right["p2_regret"]:.3f}' + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def table_depth(result: dict) -> str:
    methods = (
        ("interp", "INTERP"),
        ("p1_affine", "$p=1$ affine"),
        ("p1_feature_affine", "$p=1$ + features affine"),
    )
    lines = [
        r"\begin{table}[t]",
        r"\caption{\label{tab:depth}True depth-transfer controls, which receive each test graph's $p=1$ reference. MSE covers the four normalized $p=2$ coordinates.}",
        r"\centering\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & $p=2$ MSE & $R_2$ (\%)\\",
        r"\midrule",
    ]
    for key, label in methods:
        left = result["protocols"]["source_leave_one_out"]["aggregate_depth"][key]
        right = result["protocols"]["signature_grouped"]["aggregate_depth"][key]
        lines.append(
            f'{label} (source) & {left["p2_schedule_mse"]:.3e} & {100*left["p2_regret"]:.3f}' + r"\\"
        )
        lines.append(
            f'{label} (signature) & {right["p2_schedule_mse"]:.3e} & {100*right["p2_regret"]:.3f}' + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_operator(result: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{\label{tab:operator}Gaussian hierarchy audit at $n=144$, leaf size 16. Error is relative Frobenius error; storage is the fraction of $n^2$ scalars. Certified rows satisfy $\epsilon_i<s_i$ in Proposition~\ref{prop:ratio}.}",
        r"\centering\small",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Rank & Rel. error & Storage & Certified & Violations\\",
        r"\midrule",
    ]
    for row in result["operator_audit"]["rank_audit"]:
        lines.append(
            f'{row["rank"]} & {row["relative_frobenius_error"]:.2e} & '
            f'{100*row["storage_fraction"]:.1f}\\% & {row["certified_rows"]}/144 & '
            f'{row["prediction_bound_violations"]}' + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_tables(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "macros.tex": tex_macros(result),
        "tab_data.tex": table_data(result),
        "tab_protocols.tex": table_protocols(result),
        "tab_results.tex": table_results(result),
        "tab_depth.tex": table_depth(result),
        "tab_operator.tex": table_operator(result),
    }
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8")


def write_figures(result: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.figsize": (5.4, 3.3),
            "savefig.bbox": "tight",
        }
    )
    blue, orange, green, red, gray = "#2F6B9A", "#D17A22", "#3E8E6B", "#A6413C", "#555555"

    curve = result["collision"]["curve"]
    x = np.asarray([row["test_probability"] for row in curve])
    theory = np.asarray([row["theory"] for row in curve])
    empirical = np.asarray([row["simulation"] for row in curve])
    error = np.asarray([2 * row["simulation_standard_error"] for row in curve])
    fig, ax = plt.subplots()
    ax.plot(x, theory, "-", color=blue, label="exact collision law")
    ax.errorbar(x, empirical, yerr=error, fmt="o", color=orange, label="20,000 trials (2 s.e.)")
    ax.set(xlabel="independent test probability $\\tau$", ylabel="test-record collision probability", ylim=(-0.03, 1.03))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(output / "collision_law.pdf")
    fig.savefig(output / "collision_law.png", dpi=180)
    plt.close(fig)

    methods = ["constant", "ridge", "weighted_knn", "dense_residual", "hierarchical_residual"]
    labels = ["constant", "ridge", "nearest", "dense", "hier."]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    for protocol, color, marker, label in (
        ("source_leave_one_out", blue, "o", "source LOSO"),
        ("signature_grouped", orange, "s", "signature grouped"),
    ):
        values = [result["protocols"][protocol]["aggregate"][method]["schedule_mse"] for method in methods]
        regrets = [result["protocols"][protocol]["aggregate"][method]["p2_regret"] for method in methods]
        axes[0].plot(labels, values, marker=marker, color=color, label=label)
        axes[1].plot(labels, regrets, marker=marker, color=color, label=label)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel("normalized schedule MSE")
    axes[1].set_ylabel("mean $p=2$ relative regret")
    for ax in axes:
        ax.tick_params(axis="x", rotation=24)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "protocol_performance.pdf")
    fig.savefig(output / "protocol_performance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    for ax, protocol, title in zip(
        axes,
        ("source_leave_one_out", "signature_grouped"),
        ("held-out source", "held-out signature groups"),
    ):
        folds = result["protocols"][protocol]["folds"]
        fold_x = np.arange(len(folds))
        for method, color, marker, label in (
            ("weighted_knn", blue, "o", "nearest donor"),
            ("hierarchical_residual", red, "s", "hierarchical residual"),
        ):
            ax.plot(fold_x, [fold["methods"][method]["p2_regret"] for fold in folds], marker=marker, color=color, label=label)
        ax.plot(fold_x, [fold["depth_methods"]["p1_feature_affine"]["p2_regret"] for fold in folds], marker="^", color=green, label="$p=1$ + features")
        ax.set(title=title, xlabel="fold", ylabel="$p=2$ relative regret", yscale="symlog", ylim=(-2e-4, None))
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(output / "fold_regret.pdf")
    fig.savefig(output / "fold_regret.png", dpi=180)
    plt.close(fig)

    # Depth lifting is the primary estimand of the rewritten manuscript.  The
    # three panels deliberately expose protocol dependence, fold variation,
    # and the conservatism of the analytical energy certificate.
    depth_methods = ("interp", "p1_affine", "p1_feature_affine")
    depth_labels = ("INTERP", "$p=1$ affine", "$p=1$ + features")
    depth_colors = (gray, blue, green)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15))
    xloc = np.arange(len(depth_methods))
    width = 0.34
    for offset, protocol, label, hatch in (
        (-width / 2, "source_leave_one_out", "source held out", ""),
        (width / 2, "signature_grouped", "signature held out", "//"),
    ):
        vals = [100 * result["protocols"][protocol]["aggregate_depth"][method]["p2_regret"] for method in depth_methods]
        axes[0].bar(xloc + offset, vals, width, color=depth_colors, alpha=0.86,
                    edgecolor="white", hatch=hatch, label=label)
    axes[0].set_xticks(xloc, depth_labels, rotation=21, ha="right")
    axes[0].set_ylabel("mean depth-two regret (%)")
    axes[0].set_title("protocol dependence")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=7.2)

    folds = result["protocols"]["signature_grouped"]["folds"]
    fold_x = np.arange(len(folds))
    for method, label, color, marker in zip(depth_methods, depth_labels, depth_colors, ("o", "s", "^")):
        vals = [100 * fold["depth_methods"][method]["p2_regret"] for fold in folds]
        axes[1].plot(fold_x, vals, marker=marker, color=color, label=label)
    axes[1].set(xlabel="held-out signature fold", ylabel="depth-two regret (%)", title="fold variation")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=7.2)

    points = result["qaoa_stability"]["points"]
    observed = np.asarray([max(row["absolute_relative_energy_gap"], 1e-7) for row in points])
    bound = np.asarray([row["lipschitz_bound"] for row in points])
    lo = min(observed.min(), bound.min())
    hi = max(observed.max(), bound.max())
    axes[2].loglog(observed, bound, "o", ms=3.2, alpha=0.44, color=orange)
    axes[2].loglog([lo, hi], [lo, hi], "--", color=gray, lw=1, label="equality")
    axes[2].set(xlabel="observed relative gap", ylabel="commutator bound", title="energy certificate")
    axes[2].grid(alpha=0.22, which="both")
    axes[2].legend(frameon=False, fontsize=7.2)
    fig.tight_layout()
    fig.savefig(output / "depth_lifting.pdf")
    fig.savefig(output / "depth_lifting.png", dpi=180)
    plt.close(fig)

    ranks = result["operator_audit"]["rank_audit"]
    scaling = result["operator_audit"]["scaling"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
    rank_x = [row["rank"] for row in ranks]
    axes[0].semilogy(rank_x, [row["relative_frobenius_error"] for row in ranks], "o-", color=blue, label="kernel error")
    axes[0].semilogy(rank_x, [row["max_actual_prediction_error"] for row in ranks], "s-", color=green, label="prediction error")
    axes[0].semilogy(rank_x, [row["max_finite_prediction_bound"] for row in ranks], "^-", color=orange, label="finite bound")
    axes[0].set(xlabel="off-diagonal rank", ylabel="error or bound")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=7.5)
    size_x = [row["records"] for row in scaling]
    axes[1].plot(size_x, [1e6*row["dense_apply_seconds"] for row in scaling], "o-", color=blue, label="dense BLAS")
    axes[1].plot(size_x, [1e6*row["hierarchy_apply_seconds"] for row in scaling], "s-", color=red, label="Python hierarchy")
    axes[1].set(xlabel="real-data records", ylabel="median application time ($\\mu$s)")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(output / "operator_audit.pdf")
    fig.savefig(output / "operator_audit.png", dpi=180)
    plt.close(fig)

    points = result["qaoa_stability"]["points"]
    actual = np.maximum([point["absolute_relative_energy_gap"] for point in points], 1e-14)
    bound = np.maximum([point["lipschitz_bound"] for point in points], 1e-14)
    fig, ax = plt.subplots()
    ax.loglog(actual, bound, "o", markersize=3.5, alpha=0.65, color=blue)
    lower = min(float(np.min(actual)), float(np.min(bound)))
    upper = max(float(np.max(actual)), float(np.max(bound)))
    ax.loglog([lower, upper], [lower, upper], "--", color=gray, label="equality")
    ax.set(xlabel="observed absolute relative energy gap", ylabel="commutator Lipschitz bound")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False)
    fig.savefig(output / "stability_certificate.pdf")
    fig.savefig(output / "stability_certificate.png", dpi=180)
    plt.close(fig)


def write_outputs(result: dict) -> None:
    if CHECKOUT is None:
        raise RuntimeError("--write is available only from a source checkout")
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    for directory in (CHECKOUT / "results", PACKAGE_DIR / "results"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "results.json").write_text(payload, encoding="utf-8")
    write_tables(result, CHECKOUT / "manuscript_assets" / "tables")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args(argv)
    result = run_audit(resolve_data_root(arguments.data_dir))
    if arguments.write:
        write_outputs(result)
    if arguments.figures:
        destination = arguments.output_dir
        if destination is None:
            if CHECKOUT is None:
                raise SystemExit("installed use requires --output-dir with --figures")
            destination = CHECKOUT / "manuscript_assets" / "figures"
        write_figures(result, destination)
    if arguments.verify:
        verify(result)
        print("verified collision-aware QAOA audit")
    elif not arguments.write:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
