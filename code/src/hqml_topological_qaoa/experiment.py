"""Locked CPU reproduction for matrix-free, multi-depth QAOA transfer.

The experiment derives 256 deterministic ten-vertex induced neighborhoods from
the public Stanford Gset G1 graph.  It learns normalized p=1 and p=2 schedules
with a rank-counterfactual hierarchical kernel.  The p=2 labels are explicitly
coordinate-refined references, not proven global optima.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from .matrix_free import (
    hierarchical_prior_bundle,
    matrix_free_certificate,
)

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "results").is_dir() else SOURCE_ROOT
SEED = 260504523
G1_SHA256 = "73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gset(path: Path) -> tuple[int, list[tuple[int, int, float]]]:
    with path.open(encoding="utf-8") as handle:
        n, declared = map(int, handle.readline().split()[:2])
        edges = []
        for line in handle:
            u, v, weight = line.split()
            edges.append((int(u) - 1, int(v) - 1, float(weight)))
    if len(edges) != declared:
        raise ValueError(f"G1 declares {declared} edges but contains {len(edges)}")
    return n, edges


def bfs_vertices(adjacency: list[list[int]], start: int, size: int = 10) -> list[int]:
    seen = {start}
    queue = [start]
    for vertex in queue:
        for neighbor in adjacency[vertex]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
                if len(queue) == size:
                    return queue
    raise ValueError(f"could not obtain a connected {size}-vertex neighborhood")


def cost_diagonal(edges: list[tuple[int, int]], n: int) -> np.ndarray:
    states = np.arange(1 << n, dtype=np.uint32)
    diagonal = np.zeros(len(states), dtype=float)
    for u, v in edges:
        diagonal += ((states >> u) ^ (states >> v)) & 1
    return diagonal


def apply_mixer(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    result = state
    cosine, sine = math.cos(beta), -1j * math.sin(beta)
    for qubit in range(n):
        width = 1 << qubit
        tensor = result.reshape(-1, 2, width)
        left = tensor[:, 0, :].copy()
        right = tensor[:, 1, :].copy()
        tensor[:, 0, :] = cosine * left + sine * right
        tensor[:, 1, :] = sine * left + cosine * right
        result = tensor.reshape(-1)
    return result


def schedule_energy(
    diagonal: np.ndarray,
    gammas: np.ndarray | list[float],
    betas: np.ndarray | list[float],
    n: int,
) -> float:
    state = np.full(len(diagonal), 1.0 / math.sqrt(len(diagonal)), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp(-1j * float(gamma) * diagonal)
        state = apply_mixer(state, float(beta), n)
    return float(np.dot(np.abs(state) ** 2, diagonal))


def p1_closed_form(edges: list[tuple[int, int]], gamma: float, beta: float, n: int) -> float:
    adjacency = [set() for _ in range(n)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    cosine_gamma = math.cos(gamma)
    cosine_2gamma = math.cos(2.0 * gamma)
    sine_4beta = math.sin(4.0 * beta)
    sine_2beta = math.sin(2.0 * beta)
    total = 0.0
    for u, v in edges:
        du = len(adjacency[u]) - 1
        dv = len(adjacency[v]) - 1
        triangles = len(adjacency[u] & adjacency[v])
        first = (
            0.25
            * sine_4beta
            * math.sin(gamma)
            * (cosine_gamma**du + cosine_gamma**dv)
        )
        second = (
            0.25
            * sine_2beta**2
            * cosine_gamma ** (du + dv - 2 * triangles)
            * (1.0 - cosine_2gamma**triangles)
        )
        total += 0.5 + first - second
    return float(total)


def p1_grid_reference(edges: list[tuple[int, int]], n: int) -> tuple[np.ndarray, float]:
    gamma_grid = np.linspace(0.05, math.pi - 0.05, 13)
    beta_grid = np.linspace(0.03, math.pi / 2.0 - 0.03, 9)
    best_value = -math.inf
    best_angles = np.zeros(2)
    for gamma in gamma_grid:
        for beta in beta_grid:
            value = p1_closed_form(edges, float(gamma), float(beta), n)
            if value > best_value:
                best_value = value
                best_angles[:] = (gamma, beta)
    return best_angles, float(best_value)


def canonicalize(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    result[:2] = np.mod(result[:2], math.pi)
    result[2:] = np.clip(result[2:], 0.0, math.pi / 2.0)
    return result


def p2_refined_reference(
    diagonal: np.ndarray, p1_angles: np.ndarray, n: int
) -> tuple[np.ndarray, float, int]:
    """Return a deterministic local p=2 reference and its objective value.

    INTERP duplicates half of the p=1 angles.  Three coordinate sweeps then
    probe +/- step on four coordinates.  This is deliberately modest and fully
    auditable; the manuscript never calls it a global optimum.
    """

    vector = np.array(
        [
            p1_angles[0] / 2.0,
            p1_angles[0] / 2.0,
            p1_angles[1] / 2.0,
            p1_angles[1] / 2.0,
        ]
    )
    value = schedule_energy(diagonal, vector[:2], vector[2:], n)
    queries = 1
    for step in (0.30, 0.15, 0.075):
        for coordinate in range(4):
            candidates = []
            for sign in (-1.0, 1.0):
                trial = vector.copy()
                trial[coordinate] += sign * step
                trial = canonicalize(trial)
                trial_value = schedule_energy(diagonal, trial[:2], trial[2:], n)
                queries += 1
                candidates.append((trial_value, trial))
            proposal_value, proposal = max(candidates, key=lambda item: item[0])
            if proposal_value > value:
                value, vector = float(proposal_value), proposal
    return vector, float(value), queries


def graph_features(edges: list[tuple[int, int]], n: int) -> np.ndarray:
    adjacency = np.zeros((n, n), dtype=float)
    for u, v in edges:
        adjacency[u, v] = adjacency[v, u] = 1.0
    degrees = adjacency.sum(axis=1)
    laplacian = np.diag(degrees) - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    triangles = np.trace(adjacency @ adjacency @ adjacency) / 6.0
    wedges = float(np.sum(degrees * np.maximum(degrees - 1.0, 0.0)) / 2.0)
    return np.array(
        [
            1.0,
            len(edges) / (n * (n - 1) / 2.0),
            degrees.mean(),
            degrees.std(),
            degrees.min(),
            degrees.max(),
            triangles,
            wedges,
            eigenvalues[1],
            eigenvalues[-1],
        ],
        dtype=float,
    )


def encode_schedule(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return np.array(
        [
            p1[0] / math.pi,
            p1[1] / (math.pi / 2.0),
            p2[0] / math.pi,
            p2[1] / math.pi,
            p2[2] / (math.pi / 2.0),
            p2[3] / (math.pi / 2.0),
        ]
    )


def decode_schedule(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(np.asarray(encoded, dtype=float), 0.0, 1.0)
    p1 = np.array([clipped[0] * math.pi, clipped[1] * math.pi / 2.0])
    p2 = np.array(
        [
            clipped[2] * math.pi,
            clipped[3] * math.pi,
            clipped[4] * math.pi / 2.0,
            clipped[5] * math.pi / 2.0,
        ]
    )
    return p1, p2


def ridge_fit_predict(
    features: np.ndarray,
    targets: np.ndarray,
    fit: np.ndarray,
    predict: np.ndarray,
    penalty: float,
) -> np.ndarray:
    design_fit = np.column_stack((np.ones(len(fit)), features[fit]))
    design_predict = np.column_stack((np.ones(len(predict)), features[predict]))
    regularizer = np.eye(design_fit.shape[1])
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design_fit.T @ design_fit + penalty * regularizer,
        design_fit.T @ targets[fit],
    )
    return np.clip(design_predict @ coefficients, 0.0, 1.0)


def select_residual_correction(
    truth: np.ndarray,
    baseline: np.ndarray,
    residual_low: np.ndarray,
    residual_high: np.ndarray,
    *,
    sampled_operator_error: float,
    storage_fraction: float,
    error_tolerance: float = 0.01,
    storage_budget: float = 0.35,
) -> dict[str, float]:
    """Select the residual-correction weight without consulting test labels."""

    scale = max(float(np.mean((truth - truth.mean(axis=0)) ** 2)), 1e-8)
    best = None
    for alpha in (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0):
        high = np.clip(baseline + alpha * residual_high, 0.0, 1.0)
        low = np.clip(baseline + alpha * residual_low, 0.0, 1.0)
        data_loss = float(np.mean((high - truth) ** 2))
        counterfactual = float(np.mean((high - low) ** 2) / scale)
        certificate_violation = float(
            alpha**2
            * (
                max(0.0, sampled_operator_error - error_tolerance)
                / error_tolerance
            )
            ** 2
        )
        budget_violation = float(
            alpha**2
            * (
                max(0.0, storage_fraction - storage_budget)
                / storage_budget
            )
            ** 2
        )
        total = (
            data_loss
            + 0.25 * counterfactual
            + 0.25 * certificate_violation
            + 0.25 * budget_violation
        )
        candidate = {
            "alpha": alpha,
            "data_loss": data_loss,
            "rank_counterfactual_term": counterfactual,
            "certificate_violation": certificate_violation,
            "budget_violation": budget_violation,
            "total_loss": total,
        }
        if best is None or candidate["total_loss"] < best["total_loss"]:
            best = candidate
    assert best is not None
    return best


def build_dataset() -> dict:
    source = PACKAGE_ROOT / "data" / "source" / "G1"
    if sha256(source) != G1_SHA256:
        raise ValueError("Stanford Gset G1 SHA-256 mismatch")
    n_source, weighted_edges = parse_gset(source)
    adjacency = [[] for _ in range(n_source)]
    for u, v, _ in weighted_edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbors in adjacency:
        neighbors.sort()

    feature_rows = []
    target_rows = []
    diagonals = []
    edge_sets = []
    p1_values = []
    p2_values = []
    p2_queries = []
    for sample in range(256):
        vertices = bfs_vertices(adjacency, (sample * 37) % n_source)
        local_index = {vertex: index for index, vertex in enumerate(vertices)}
        local_edges = [
            (local_index[u], local_index[v])
            for u in vertices
            for v in adjacency[u]
            if u < v and v in local_index
        ]
        diagonal = cost_diagonal(local_edges, len(vertices))
        p1_angles, p1_value = p1_grid_reference(local_edges, len(vertices))
        p2_angles, p2_value, queries = p2_refined_reference(
            diagonal, p1_angles, len(vertices)
        )
        feature_rows.append(graph_features(local_edges, len(vertices)))
        target_rows.append(encode_schedule(p1_angles, p2_angles))
        diagonals.append(diagonal)
        edge_sets.append(local_edges)
        p1_values.append(p1_value)
        p2_values.append(p2_value)
        p2_queries.append(queries)
    return {
        "features": np.asarray(feature_rows),
        "targets": np.asarray(target_rows),
        "diagonals": diagonals,
        "edge_sets": edge_sets,
        "p1_values": np.asarray(p1_values),
        "p2_values": np.asarray(p2_values),
        "p2_queries": np.asarray(p2_queries),
        "source_vertices": n_source,
        "source_edges": len(weighted_edges),
    }


def task_regret(
    predictions: np.ndarray,
    indices: np.ndarray,
    data: dict,
) -> tuple[float, float]:
    p1_regrets = []
    p2_regrets = []
    for row, index in enumerate(indices):
        p1, p2 = decode_schedule(predictions[row])
        p1_achieved = p1_closed_form(
            data["edge_sets"][index], p1[0], p1[1], 10
        )
        p2_achieved = schedule_energy(
            data["diagonals"][index], p2[:2], p2[2:], 10
        )
        p1_regrets.append(
            max(
                0.0,
                (data["p1_values"][index] - p1_achieved)
                / max(data["p1_values"][index], 1e-12),
            )
        )
        p2_regrets.append(
            max(
                0.0,
                (data["p2_values"][index] - p2_achieved)
                / max(data["p2_values"][index], 1e-12),
            )
        )
    return float(np.mean(p1_regrets)), float(np.mean(p2_regrets))


def case_metrics(
    ridge_predictions: np.ndarray,
    rcol_predictions: np.ndarray,
    indices: np.ndarray,
    data: dict,
) -> list[dict]:
    rows = []
    for row, index in enumerate(indices):
        ridge_p1, ridge_p2 = decode_schedule(ridge_predictions[row])
        rcol_p1, rcol_p2 = decode_schedule(rcol_predictions[row])
        ridge_value_p1 = p1_closed_form(
            data["edge_sets"][index], ridge_p1[0], ridge_p1[1], 10
        )
        rcol_value_p1 = p1_closed_form(
            data["edge_sets"][index], rcol_p1[0], rcol_p1[1], 10
        )
        ridge_value_p2 = schedule_energy(
            data["diagonals"][index], ridge_p2[:2], ridge_p2[2:], 10
        )
        rcol_value_p2 = schedule_energy(
            data["diagonals"][index], rcol_p2[:2], rcol_p2[2:], 10
        )
        reference_p1 = data["p1_values"][index]
        reference_p2 = data["p2_values"][index]
        rows.append(
            {
                "index": int(index),
                "edges": len(data["edge_sets"][index]),
                "ridge_schedule_mse": float(
                    np.mean((ridge_predictions[row] - data["targets"][index]) ** 2)
                ),
                "rcol_schedule_mse": float(
                    np.mean((rcol_predictions[row] - data["targets"][index]) ** 2)
                ),
                "ridge_regret_p1": float(
                    max(0.0, (reference_p1 - ridge_value_p1) / reference_p1)
                ),
                "rcol_regret_p1": float(
                    max(0.0, (reference_p1 - rcol_value_p1) / reference_p1)
                ),
                "ridge_regret_p2": float(
                    max(0.0, (reference_p2 - ridge_value_p2) / reference_p2)
                ),
                "rcol_regret_p2": float(
                    max(0.0, (reference_p2 - rcol_value_p2) / reference_p2)
                ),
            }
        )
    return rows


def run_experiment() -> dict:
    started = time.perf_counter()
    data = build_dataset()
    features = data["features"]
    targets = data["targets"]
    indices = np.arange(len(features))
    test = indices[indices % 4 == 0]
    train = indices[indices % 4 != 0]
    development = train[::5]
    fit = np.setdiff1d(train, development)

    mean = features[fit].mean(axis=0)
    scale = features[fit].std(axis=0)
    active = scale >= 1e-12
    if not np.any(active):
        raise ValueError("all graph features are constant on the fit split")
    standardized = (features[:, active] - mean[active]) / scale[active]

    penalties = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    penalty_losses = {}
    for penalty in penalties:
        prediction = ridge_fit_predict(
            standardized, targets, fit, development, penalty
        )
        penalty_losses[str(penalty)] = float(
            np.mean((prediction - targets[development]) ** 2)
        )
    selected_penalty = min(penalties, key=lambda value: penalty_losses[str(value)])
    ridge_fit_all = ridge_fit_predict(
        standardized, targets, fit, indices, selected_penalty
    )
    ridge_train_all = ridge_fit_predict(
        standardized, targets, train, indices, selected_penalty
    )
    ridge_development = ridge_fit_all[development]
    ridge_test = ridge_train_all[test]

    fit_mask = np.zeros(len(indices), dtype=bool)
    fit_mask[fit] = True
    train_mask = np.zeros(len(indices), dtype=bool)
    train_mask[train] = True
    residual_fit = targets - ridge_fit_all
    residual_train = targets - ridge_train_all
    bandwidth_candidates = (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12)
    bandwidth_path = {}
    development_priors = {}
    for bandwidth in bandwidth_candidates:
        candidate_prior = hierarchical_prior_bundle(
            standardized,
            [fit_mask],
            residual_fit,
            SEED,
            bandwidth=bandwidth,
            leaf_size=16,
            low_rank=2,
            high_rank=8,
        )
        candidate = select_residual_correction(
            targets[development],
            ridge_development,
            candidate_prior["low"][0][development],
            candidate_prior["high"][0][development],
            sampled_operator_error=candidate_prior["metrics"][
                "hmat_sampled_row_relative_error"
            ],
            storage_fraction=candidate_prior["metrics"]["hmat_storage_fraction"],
        )
        bandwidth_path[str(bandwidth)] = candidate
        development_priors[bandwidth] = candidate_prior
    selected_bandwidth = min(
        bandwidth_candidates,
        key=lambda value: bandwidth_path[str(value)]["total_loss"],
    )
    selected = bandwidth_path[str(selected_bandwidth)]
    prior_development = development_priors[selected_bandwidth]
    prior_test = hierarchical_prior_bundle(
        standardized,
        [train_mask],
        residual_train,
        SEED,
        bandwidth=selected_bandwidth,
        leaf_size=16,
        low_rank=2,
        high_rank=8,
    )
    alpha = selected["alpha"]
    rcol_test = np.clip(
        ridge_test + alpha * prior_test["high"][0][test],
        0.0,
        1.0,
    )
    low_test = np.clip(
        ridge_test + alpha * prior_test["low"][0][test],
        0.0,
        1.0,
    )
    constant_test = np.tile(targets[train].mean(axis=0), (len(test), 1))

    constant_loss = float(np.mean((constant_test - targets[test]) ** 2))
    ridge_loss = float(np.mean((ridge_test - targets[test]) ** 2))
    rcol_loss = float(np.mean((rcol_test - targets[test]) ** 2))
    constant_regret_p1, constant_regret_p2 = task_regret(constant_test, test, data)
    ridge_regret_p1, ridge_regret_p2 = task_regret(ridge_test, test, data)
    rcol_regret_p1, rcol_regret_p2 = task_regret(rcol_test, test, data)
    rank_counterfactual = float(
        np.mean((rcol_test - low_test) ** 2)
        / max(np.mean((targets[test] - targets[train].mean(axis=0)) ** 2), 1e-12)
    )
    test_cases = case_metrics(ridge_test, rcol_test, test, data)

    _, source_edges = parse_gset(PACKAGE_ROOT / "data" / "source" / "G1")
    source_degrees = np.zeros(data["source_vertices"], dtype=float)
    for u, v, _ in source_edges:
        source_degrees[u] += 1.0
        source_degrees[v] += 1.0
    full_graph_features = np.column_stack(
        (
            source_degrees,
            np.arange(data["source_vertices"], dtype=float)
            / data["source_vertices"],
        )
    )
    scaling_certificate = matrix_free_certificate(
        full_graph_features,
        SEED + 1,
        bandwidth=0.08,
        leaf_size=32,
        low_rank=2,
        high_rank=8,
    )

    return {
        "schema_version": 2,
        "package": "hqml_topological_qaoa",
        "manuscript_key": "topological_qaoa",
        "seed": SEED,
        "dataset": "Stanford Gset G1",
        "dataset_url": "https://web.stanford.edu/~yyye/yyye/Gset/G1",
        "dataset_sha256": G1_SHA256,
        "source_vertices": data["source_vertices"],
        "source_edges": data["source_edges"],
        "learning_instances": len(features),
        "train_instances": len(train),
        "development_instances": len(development),
        "test_instances": len(test),
        "raw_feature_dimension": features.shape[1],
        "feature_dimension": standardized.shape[1],
        "target_dimension": targets.shape[1],
        "p1_grid_queries": 13 * 9,
        "p2_refinement_queries": int(data["p2_queries"][0]),
        "selected_ridge_penalty": selected_penalty,
        "ridge_development_losses": penalty_losses,
        "selected_kernel_bandwidth": selected_bandwidth,
        "kernel_bandwidth_path": bandwidth_path,
        "rcol_selected_alpha": alpha,
        "rcol_development_data_loss": selected["data_loss"],
        "rcol_development_rank_counterfactual_term": selected[
            "rank_counterfactual_term"
        ],
        "rcol_development_total_loss": selected["total_loss"],
        "rcol_test_rank_counterfactual_term": rank_counterfactual,
        "constant_schedule_mse": constant_loss,
        "ridge_schedule_mse": ridge_loss,
        "rcol_schedule_mse": rcol_loss,
        "constant_regret_p1": constant_regret_p1,
        "ridge_regret_p1": ridge_regret_p1,
        "rcol_regret_p1": rcol_regret_p1,
        "constant_regret_p2": constant_regret_p2,
        "ridge_regret_p2": ridge_regret_p2,
        "rcol_regret_p2": rcol_regret_p2,
        "test_cases": test_cases,
        "matrix_free": prior_test["metrics"]["matrix_free"],
        "training_global_dense_matrix_materialized": prior_test["metrics"][
            "global_dense_matrix_materialized"
        ],
        **{f"training_{key}": value for key, value in prior_test["metrics"].items()},
        **{f"scaling_{key}": value for key, value in scaling_certificate.items()},
        "cpu_seconds": time.perf_counter() - started,
    }


def rounded(result: dict) -> dict:
    def convert(value):
        if isinstance(value, float):
            return round(value, 12)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(result)


def results_tex(result: dict) -> str:
    macros = {
        "LearningInstances": result["learning_instances"],
        "TrainInstances": result["train_instances"],
        "DevelopmentInstances": result["development_instances"],
        "TestInstances": result["test_instances"],
        "ConstantMSE": f'{result["constant_schedule_mse"]:.6f}',
        "RidgeMSE": f'{result["ridge_schedule_mse"]:.6f}',
        "RCOLMSE": f'{result["rcol_schedule_mse"]:.6f}',
        "RidgeRegretOne": f'{100 * result["ridge_regret_p1"]:.3f}',
        "RCOLRegretOne": f'{100 * result["rcol_regret_p1"]:.3f}',
        "RidgeRegretTwo": f'{100 * result["ridge_regret_p2"]:.3f}',
        "RCOLRegretTwo": f'{100 * result["rcol_regret_p2"]:.3f}',
        "RCOLAlpha": f'{result["rcol_selected_alpha"]:.3f}',
        "TrainingStorage": f'{100 * result["training_hmat_storage_fraction"]:.3f}',
        "TrainingWorkRatio": f'{result["training_hmat_matmat_work_ratio"]:.3f}',
        "TrainingError": f'{result["training_hmat_sampled_row_relative_error"]:.3e}',
        "TrainingPeak": f'{100 * result["training_hmat_peak_block_fraction"]:.3f}',
        "ScalingStorage": f'{100 * result["scaling_hmat_storage_fraction"]:.3f}',
        "ScalingWorkRatio": f'{result["scaling_hmat_matmat_work_ratio"]:.3f}',
        "CPUSeconds": f'{result["cpu_seconds"]:.2f}',
    }
    return "\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items()
    ) + "\n"


def case_tables_tex(result: dict) -> str:
    rows = result["test_cases"]
    blocks = []
    for block_index, start in enumerate(range(0, len(rows), 16), start=1):
        current = rows[start : start + 16]
        lines = [
            r"\begin{table}[t]",
            rf"\caption{{Held-out case audit, block {block_index} of 4. MSE is per six-coordinate schedule; regrets are percentages.}}",
            r"\begin{ruledtabular}",
            r"\begin{tabular}{rrrrrr}",
            r"ID & Edges & Ridge MSE & RCOL MSE & Ridge $R_2$ & RCOL $R_2$\\",
            r"\hline",
        ]
        for item in current:
            lines.append(
                f'{item["index"]} & {item["edges"]} & '
                f'{item["ridge_schedule_mse"]:.5f} & '
                f'{item["rcol_schedule_mse"]:.5f} & '
                f'{100 * item["ridge_regret_p2"]:.3f} & '
                f'{100 * item["rcol_regret_p2"]:.3f}' + r"\\"
            )
        lines.extend(
            [
                r"\end{tabular}",
                r"\end{ruledtabular}",
                r"\end{table}",
                "",
            ]
        )
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def write_figures(result: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    colors = ("#4C78A8", "#F58518", "#54A24B")
    plt.rcParams.update({"font.size": 9, "figure.figsize": (5.2, 3.2)})

    labels = ["constant", "ridge", "RCOL"]
    values = [
        result["constant_schedule_mse"],
        result["ridge_schedule_mse"],
        result["rcol_schedule_mse"],
    ]
    figure, axis = plt.subplots()
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("held-out normalized schedule MSE")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "schedule_mse_bar.pdf")
    plt.close(figure)

    x = np.arange(2)
    width = 0.36
    figure, axis = plt.subplots()
    axis.bar(
        x - width / 2,
        [100 * result["ridge_regret_p1"], 100 * result["ridge_regret_p2"]],
        width,
        label="ridge",
        color=colors[1],
    )
    axis.bar(
        x + width / 2,
        [100 * result["rcol_regret_p1"], 100 * result["rcol_regret_p2"]],
        width,
        label="RCOL",
        color=colors[2],
    )
    axis.set_xticks(x, ["depth 1", "depth 2"])
    axis.set_ylabel("mean reference-relative regret (%)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "regret_bar.pdf")
    plt.close(figure)

    penalties = np.array([float(item) for item in result["ridge_development_losses"]])
    losses = np.array(list(result["ridge_development_losses"].values()))
    order = np.argsort(penalties)
    figure, axis = plt.subplots()
    axis.semilogx(penalties[order], losses[order], marker="o", color=colors[0])
    axis.axvline(result["selected_ridge_penalty"], color="#555555", linestyle="--")
    axis.set_xlabel("ridge penalty")
    axis.set_ylabel("development schedule MSE")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "ridge_path_line.pdf")
    plt.close(figure)

    sizes = np.array([256, 800])
    storage = np.array(
        [
            result["training_hmat_storage_fraction"],
            result["scaling_hmat_storage_fraction"],
        ]
    )
    work = np.array(
        [
            result["training_hmat_matmat_work_ratio"],
            result["scaling_hmat_matmat_work_ratio"],
        ]
    )
    figure, axis_left = plt.subplots()
    axis_left.plot(
        sizes, 100 * storage, marker="o", color=colors[0], label="stored scalars"
    )
    axis_left.set_xlabel("operator records")
    axis_left.set_ylabel("storage relative to dense (%)", color=colors[0])
    axis_right = axis_left.twinx()
    axis_right.plot(
        sizes, work, marker="s", color=colors[2], label="counted work ratio"
    )
    axis_right.set_ylabel("dense / hierarchical counted work", color=colors[2])
    axis_left.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "operator_scaling_line.pdf")
    plt.close(figure)


def reproduce(*, write: bool = False, figures: bool = False) -> dict:
    result = rounded(run_experiment())
    if write:
        results_dir = PACKAGE_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        payload_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
        payload_tex = results_tex(result)
        payload_cases = case_tables_tex(result)
        for destination in (
            results_dir,
            SOURCE_ROOT / "results",
        ):
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "results.json").write_text(
                payload_json, encoding="utf-8"
            )
            (destination / "results.tex").write_text(
                payload_tex, encoding="utf-8"
            )
            (destination / "case_tables.tex").write_text(
                payload_cases, encoding="utf-8"
            )
    if figures:
        write_figures(result, PROJECT_ROOT.parent / "figures")
    return result


def verify(actual: dict) -> None:
    expected_path = PACKAGE_ROOT / "results" / "results.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    ignored = {"cpu_seconds"}
    for key in sorted(set(actual) | set(expected)):
        if key in ignored:
            continue
        if key not in actual or key not in expected:
            raise AssertionError(f"key mismatch: {key}")
        if isinstance(actual[key], float):
            absolute_tolerance = (
                1e-8 if key.endswith("sampled_row_relative_error") else 1e-10
            )
            if not math.isclose(
                actual[key],
                expected[key],
                rel_tol=1e-9,
                abs_tol=absolute_tolerance,
            ):
                raise AssertionError(f"{key}: {actual[key]} != {expected[key]}")
        elif actual[key] != expected[key]:
            raise AssertionError(f"{key}: {actual[key]} != {expected[key]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--figures", action="store_true")
    arguments = parser.parse_args(argv)
    result = reproduce(write=arguments.write, figures=arguments.figures)
    if arguments.verify:
        verify(result)
        print("verified locked QAOA results")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
