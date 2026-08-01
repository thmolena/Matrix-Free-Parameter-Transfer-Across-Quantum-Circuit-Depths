"""Matrix-free hierarchical kernels and the rank-counterfactual operator loss.

The module never constructs an n-by-n kernel.  Kernel entries are requested
only for dense leaves, cross-approximation skeletons, or streamed validation
rows.  The implementation is deliberately NumPy-only so the archival wheel
has one runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EntryStats:
    calls: int = 0
    queried_entries: int = 0
    peak_block_entries: int = 0


class KernelEntryOracle:
    """Gaussian-plus-periodic kernel exposed only through block queries."""

    def __init__(self, key: np.ndarray, bandwidth: float = 0.10):
        values = np.asarray(key, dtype=float).reshape(-1)
        span = max(float(np.ptp(values)), 1e-12)
        self.key = (values - float(values.min())) / span
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
        matrix = np.exp(-0.5 * (delta / self.bandwidth) ** 2)
        matrix += 0.08 * np.cos(2.0 * np.pi * delta)
        matrix += 0.04 * np.cos(4.0 * np.pi * delta)
        matrix += 1e-4 * (rows[:, None] == cols[None, :])
        return matrix


@dataclass
class HNode:
    n: int
    leaf: np.ndarray | None = None
    left: "HNode | None" = None
    right: "HNode | None" = None
    u01: np.ndarray | None = None
    v01: np.ndarray | None = None

    @property
    def storage(self) -> int:
        if self.leaf is not None:
            return int(self.leaf.size)
        assert self.left is not None and self.right is not None
        assert self.u01 is not None and self.v01 is not None
        return (
            self.left.storage
            + self.right.storage
            + int(self.u01.size)
            + int(self.v01.size)
        )

    def apply(self, batch: np.ndarray) -> tuple[np.ndarray, int]:
        batch = np.asarray(batch, dtype=float)
        if batch.ndim == 1:
            batch = batch[:, None]
        if self.leaf is not None:
            work = self.leaf.shape[0] * self.leaf.shape[1] * batch.shape[1]
            return self.leaf @ batch, int(work)
        assert self.left is not None and self.right is not None
        assert self.u01 is not None and self.v01 is not None
        split = self.left.n
        top, work_left = self.left.apply(batch[:split])
        bottom, work_right = self.right.apply(batch[split:])
        top += self.u01 @ (self.v01.T @ batch[split:])
        bottom += self.v01 @ (self.u01.T @ batch[:split])
        rank = self.u01.shape[1]
        cross_work = 2 * rank * batch.shape[1] * self.n
        return np.vstack((top, bottom)), int(work_left + work_right + cross_work)


def _landmarks(size: int, count: int) -> np.ndarray:
    count = min(size, count)
    if count == size:
        return np.arange(size, dtype=int)
    # Chebyshev-like spacing resolves block endpoints more strongly than a
    # uniform skeleton while remaining deterministic.
    angle = np.linspace(0.0, np.pi, count)
    raw = 0.5 * (1.0 - np.cos(angle)) * (size - 1)
    return np.unique(np.rint(raw).astype(int))


def _cross_factor(
    oracle: KernelEntryOracle,
    rows: np.ndarray,
    cols: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_landmarks = _landmarks(len(rows), rank)
    col_landmarks = _landmarks(len(cols), rank)
    skeleton_rows = rows[row_landmarks]
    skeleton_cols = cols[col_landmarks]
    columns = oracle.block(rows, skeleton_cols)
    rows_block = oracle.block(skeleton_rows, cols)
    intersection = oracle.block(skeleton_rows, skeleton_cols)
    inverse = np.linalg.pinv(intersection, rcond=1e-10)
    return columns @ inverse, rows_block.T


def build_hierarchy(
    oracle: KernelEntryOracle,
    indices: np.ndarray,
    *,
    leaf_size: int,
    rank: int,
) -> HNode:
    indices = np.asarray(indices, dtype=int)
    n = len(indices)
    if n <= leaf_size:
        return HNode(n=n, leaf=oracle.block(indices, indices))
    split = n // 2
    left_indices = indices[:split]
    right_indices = indices[split:]
    u01, v01 = _cross_factor(oracle, left_indices, right_indices, rank)
    return HNode(
        n=n,
        left=build_hierarchy(
            oracle, left_indices, leaf_size=leaf_size, rank=rank
        ),
        right=build_hierarchy(
            oracle, right_indices, leaf_size=leaf_size, rank=rank
        ),
        u01=u01,
        v01=v01,
    )


def _fwht_rows(values: np.ndarray) -> tuple[np.ndarray, int]:
    source = np.asarray(values, dtype=float)
    width = 1 << max(0, int(np.ceil(np.log2(max(source.shape[1], 1)))))
    transformed = np.zeros((len(source), width), dtype=float)
    transformed[:, : source.shape[1]] = source
    step = 1
    operations = 0
    while step < width:
        for start in range(0, width, 2 * step):
            left = transformed[:, start : start + step].copy()
            right = transformed[:, start + step : start + 2 * step].copy()
            transformed[:, start : start + step] = left + right
            transformed[:, start + step : start + 2 * step] = left - right
            operations += 2 * len(source) * step
        step *= 2
    transformed /= np.sqrt(width)
    return transformed, operations


def _butterfly_certificate(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    original = rng.normal(size=(1, 256))
    transformed, operations = _fwht_rows(original)
    recovered, _ = _fwht_rows(transformed)
    error = float(np.max(np.abs(recovered - original)))
    return {
        "butterfly_work_ratio": float(256**2 / operations),
        "butterfly_inverse_error": error,
    }


def _hierarchy_key(features: np.ndarray) -> tuple[np.ndarray, int]:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-10] = 1.0
    standardized = (matrix - center) / scale
    transformed, operations = _fwht_rows(standardized)
    covariance = transformed.T @ transformed / max(len(transformed), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    key = transformed @ direction
    # The sign of an eigenvector is arbitrary; fixing it makes platform
    # comparisons deterministic.
    first = int(np.argmax(np.abs(direction)))
    if direction[first] < 0:
        key = -key
    return key, operations


def _streamed_exact_rows(
    oracle: KernelEntryOracle,
    sample_rows: np.ndarray,
    batch: np.ndarray,
    *,
    chunk_size: int = 2048,
) -> np.ndarray:
    output = np.zeros((len(sample_rows), batch.shape[1]), dtype=float)
    for start in range(0, oracle.n, chunk_size):
        stop = min(start + chunk_size, oracle.n)
        cols = np.arange(start, stop, dtype=int)
        output += oracle.block(sample_rows, cols) @ batch[start:stop]
    return output


def _operator_pair(
    features: np.ndarray,
    seed: int,
    *,
    bandwidth: float,
    leaf_size: int,
    low_rank: int,
    high_rank: int,
    validation_rhs: int,
    validation_rows: int,
) -> tuple[np.ndarray, HNode, HNode, KernelEntryOracle, dict[str, float | int | bool]]:
    key, feature_butterfly_work = _hierarchy_key(features)
    order = np.argsort(key, kind="mergesort")
    sorted_key = key[order]
    indices = np.arange(len(order), dtype=int)
    oracle = KernelEntryOracle(sorted_key, bandwidth=bandwidth)
    low = build_hierarchy(
        oracle, indices, leaf_size=leaf_size, rank=low_rank
    )
    high = build_hierarchy(
        oracle, indices, leaf_size=leaf_size, rank=high_rank
    )

    rng = np.random.default_rng(seed)
    batch = rng.normal(size=(len(order), validation_rhs))
    approximate, work = high.apply(batch)
    row_count = min(validation_rows, max(1, len(order) // 4))
    sample_rows = np.unique(
        np.linspace(0, len(order) - 1, row_count).round().astype(int)
    )
    exact = _streamed_exact_rows(
        oracle,
        sample_rows,
        batch,
        chunk_size=min(2048, max(1, len(order) // 4)),
    )
    denominator = max(float(np.linalg.norm(exact)), 1e-15)
    error = float(np.linalg.norm(exact - approximate[sample_rows]) / denominator)
    dense_work = len(order) ** 2 * validation_rhs
    butterfly = _butterfly_certificate(seed + 7919)
    metrics: dict[str, float | int | bool] = {
        "matrix_free": True,
        "global_dense_matrix_materialized": False,
        "operator_records": int(len(order)),
        "hmat_rank_low": int(low_rank),
        "hmat_rank_high": int(high_rank),
        "hmat_sampled_rows": int(len(sample_rows)),
        "hmat_sampled_row_relative_error": error,
        "hmat_storage_fraction": float(high.storage / max(len(order) ** 2, 1)),
        "hmat_matmat_work_ratio": float(dense_work / max(work, 1)),
        "hmat_entry_query_fraction": float(
            oracle.stats.queried_entries / max(len(order) ** 2, 1)
        ),
        "hmat_peak_block_fraction": float(
            oracle.stats.peak_block_entries / max(len(order) ** 2, 1)
        ),
        "hmat_peak_block_entries": int(oracle.stats.peak_block_entries),
        "hmat_entry_query_calls": int(oracle.stats.calls),
        "feature_butterfly_work": int(feature_butterfly_work),
        **butterfly,
    }
    return order, low, high, oracle, metrics


def hierarchical_prior_bundle(
    features: np.ndarray,
    known_masks: list[np.ndarray],
    targets: np.ndarray,
    seed: int,
    *,
    bandwidth: float = 0.10,
    leaf_size: int = 32,
    low_rank: int = 2,
    high_rank: int = 8,
) -> dict:
    """Return low/high-rank predictions for several supervision masks."""

    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets[:, None]
    order, low, high, _, metrics = _operator_pair(
        features,
        seed,
        bandwidth=bandwidth,
        leaf_size=leaf_size,
        low_rank=low_rank,
        high_rank=high_rank,
        validation_rhs=min(4, targets.shape[1] + 1),
        validation_rows=64,
    )
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    low_predictions: list[np.ndarray] = []
    high_predictions: list[np.ndarray] = []
    work_low = 0
    work_high = 0
    for raw_mask in known_masks:
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (len(targets),):
            raise ValueError("known mask has the wrong shape")
        fallback = np.mean(targets[mask], axis=0)
        payload = np.column_stack((targets * mask[:, None], mask.astype(float)))
        payload = payload[order]
        low_output, current_low = low.apply(payload)
        high_output, current_high = high.apply(payload)
        work_low += current_low
        work_high += current_high
        denominator_low = low_output[:, -1:]
        denominator_high = high_output[:, -1:]
        prediction_low = np.divide(
            low_output[:, :-1],
            denominator_low,
            out=np.tile(fallback, (len(targets), 1)),
            where=np.abs(denominator_low) > 1e-8,
        )
        prediction_high = np.divide(
            high_output[:, :-1],
            denominator_high,
            out=np.tile(fallback, (len(targets), 1)),
            where=np.abs(denominator_high) > 1e-8,
        )
        low_predictions.append(prediction_low[inverse])
        high_predictions.append(prediction_high[inverse])
    metrics["prior_low_apply_work"] = int(work_low)
    metrics["prior_high_apply_work"] = int(work_high)
    return {
        "low": low_predictions,
        "high": high_predictions,
        "metrics": metrics,
    }


def select_rank_counterfactual_blend(
    truth: np.ndarray,
    baseline: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    loss: str = "mse",
    consistency_weight: float = 0.25,
    sampled_operator_error: float = 0.0,
    error_tolerance: float = 0.01,
    storage_fraction: float = 0.0,
    storage_budget: float = 0.75,
    certificate_weight: float = 0.25,
    budget_weight: float = 0.25,
) -> dict[str, float]:
    """Select a prior weight with the rank-counterfactual operator loss."""

    truth = np.asarray(truth, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    prior_low = np.asarray(prior_low, dtype=float)
    prior_high = np.asarray(prior_high, dtype=float)
    if truth.ndim == 1:
        truth = truth[:, None]
        baseline = baseline[:, None]
        prior_low = prior_low[:, None]
        prior_high = prior_high[:, None]
    if weights is None:
        weights = np.full(len(truth), 1.0 / max(len(truth), 1))
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / max(float(weights.sum()), 1e-15)
    scale = max(float(np.average(np.sum((truth - truth.mean(axis=0)) ** 2, axis=1), weights=weights)), 1e-8)

    best: dict[str, float] | None = None
    for alpha in (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0):
        prediction_high = (1.0 - alpha) * baseline + alpha * prior_high
        prediction_low = (1.0 - alpha) * baseline + alpha * prior_low
        if loss == "log":
            clipped = np.clip(prediction_high, 1e-8, 1.0 - 1e-8)
            row_loss = -(
                truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped)
            ).mean(axis=1)
        else:
            row_loss = np.mean((prediction_high - truth) ** 2, axis=1)
        data_loss = float(np.sum(weights * row_loss))
        shift = np.mean((prediction_high - prediction_low) ** 2, axis=1)
        counterfactual = float(np.sum(weights * shift) / scale)
        certificate_violation = float(
            alpha**2
            * (
                max(0.0, sampled_operator_error - error_tolerance)
                / max(error_tolerance, 1e-15)
            )
            ** 2
        )
        budget_violation = float(
            alpha**2
            * (
                max(0.0, storage_fraction - storage_budget)
                / max(storage_budget, 1e-15)
            )
            ** 2
        )
        total = (
            data_loss
            + consistency_weight * counterfactual
            + certificate_weight * certificate_violation
            + budget_weight * budget_violation
        )
        candidate = {
            "alpha": float(alpha),
            "data_loss": data_loss,
            "rank_counterfactual_term": counterfactual,
            "certificate_violation": certificate_violation,
            "budget_violation": budget_violation,
            "total_loss": float(total),
            "consistency_weight": float(consistency_weight),
            "certificate_weight": float(certificate_weight),
            "budget_weight": float(budget_weight),
            "error_tolerance": float(error_tolerance),
            "storage_budget": float(storage_budget),
        }
        if best is None or candidate["total_loss"] < best["total_loss"]:
            best = candidate
    assert best is not None
    return best


def matrix_free_certificate(
    features: np.ndarray,
    seed: int,
    *,
    bandwidth: float = 0.08,
    leaf_size: int = 32,
    low_rank: int = 2,
    high_rank: int = 8,
) -> dict[str, float | int | bool]:
    """Certify a matrix-free hierarchy without fitting a predictive prior."""

    _, _, _, _, metrics = _operator_pair(
        features,
        seed,
        bandwidth=bandwidth,
        leaf_size=leaf_size,
        low_rank=low_rank,
        high_rank=high_rank,
        validation_rhs=8,
        validation_rows=64,
    )
    return metrics
