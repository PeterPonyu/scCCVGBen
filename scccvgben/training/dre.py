# Ported from /home/zeyufu/LAB/CCVGAE/DRE.py (revised 2026-04-23)
"""Dimensionality Reduction Evaluator — Q-metrics and distance correlation.

Public entry point: ``evaluate_dimensionality_reduction(X_high, X_low, k=10)``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances


# ── ranking helpers ────────────────────────────────────────────────────────────

def get_ranking_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """Compute ranking matrix from a square distance matrix.

    Parameters
    ----------
    distance_matrix : (n, n) array of pairwise distances.

    Returns
    -------
    ranking_matrix : (n, n) int32 array where entry [i, j] is the rank of
        sample j among neighbours of sample i (0-indexed, self excluded).
    """
    try:
        n = len(distance_matrix)
        sorted_indices = np.argsort(distance_matrix, axis=1)

        ranking_matrix = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            ranking_matrix[i, sorted_indices[i]] = np.arange(n)

        mask = np.eye(n, dtype=bool)
        ranking_matrix[~mask] = ranking_matrix[~mask] - 1
        ranking_matrix[mask] = 0

        return ranking_matrix

    except Exception as exc:
        warnings.warn(f"get_ranking_matrix error: {exc}")
        return np.zeros((len(distance_matrix), len(distance_matrix)), dtype=np.int32)


def get_coranking_matrix(rank_high: np.ndarray, rank_low: np.ndarray) -> np.ndarray:
    """Compute co-ranking matrix from two ranking matrices.

    Parameters
    ----------
    rank_high : ranking matrix in the high-dimensional space.
    rank_low  : ranking matrix in the low-dimensional space.

    Returns
    -------
    corank : (n-1, n-1) int32 co-ranking matrix.
    """
    try:
        n = len(rank_high)
        corank = np.zeros((n - 1, n - 1), dtype=np.int32)

        mask = (rank_high > 0) & (rank_low > 0)
        valid_high = rank_high[mask] - 1  # convert to 0-based
        valid_low = rank_low[mask] - 1

        valid_mask = (valid_high < n - 1) & (valid_low < n - 1)
        valid_high = valid_high[valid_mask]
        valid_low = valid_low[valid_mask]

        np.add.at(corank, (valid_high, valid_low), 1)
        return corank

    except Exception as exc:
        warnings.warn(f"get_coranking_matrix error: {exc}")
        n = len(rank_high)
        return np.zeros((n - 1, n - 1), dtype=np.int32)


def compute_qnx_series(corank: np.ndarray) -> np.ndarray:
    """Compute the Q_NX quality series from a co-ranking matrix.

    Parameters
    ----------
    corank : (n-1, n-1) co-ranking matrix.

    Returns
    -------
    qnx_values : 1-D array of normalised Q_NX values for K = 1 … n-2.
    """
    try:
        n = corank.shape[0] + 1
        qnx_values = []
        qnx_cum = 0

        for K in range(1, n - 1):
            if K - 1 < corank.shape[0]:
                intrusions = int(np.sum(corank[:K, K - 1])) if K - 1 < corank.shape[1] else 0
                extrusions = int(np.sum(corank[K - 1, :K])) if K - 1 < corank.shape[0] else 0
                diagonal = int(corank[K - 1, K - 1]) if K - 1 < min(corank.shape) else 0

                qnx_cum += intrusions + extrusions - diagonal
                qnx_values.append(qnx_cum / (K * n))

        return np.array(qnx_values) if qnx_values else np.array([0.0])

    except Exception as exc:
        warnings.warn(f"compute_qnx_series error: {exc}")
        return np.array([0.0])


def get_q_local_global(
    qnx_values: np.ndarray,
) -> tuple[float, float, int]:
    """Derive Q_local, Q_global, and K_max from a Q_NX series.

    Uses the Local Continuity Meta-Criterion (LCMC) to locate K_max.

    Parameters
    ----------
    qnx_values : 1-D array from ``compute_qnx_series``.

    Returns
    -------
    (Q_local, Q_global, K_max) tuple.
    """
    try:
        if len(qnx_values) == 0:
            return 0.0, 0.0, 1

        N = len(qnx_values)
        lcmc = qnx_values.copy()
        for j in range(N):
            lcmc[j] -= j / N

        K_max = int(np.argmax(lcmc)) + 1

        Q_local = float(np.mean(qnx_values[:K_max])) if K_max > 0 else float(qnx_values[0])
        Q_global = (
            float(np.mean(qnx_values[K_max:]))
            if K_max < len(qnx_values)
            else float(qnx_values[-1])
        )
        return Q_local, Q_global, K_max

    except Exception as exc:
        warnings.warn(f"get_q_local_global error: {exc}")
        return 0.0, 0.0, 1


# ── public entry point ─────────────────────────────────────────────────────────

def evaluate_dimensionality_reduction(
    X_high: np.ndarray,
    X_low: np.ndarray,
    k: int = 10,
    verbose: bool = False,
) -> dict:
    """Evaluate quality of a dimensionality reduction.

    Parameters
    ----------
    X_high  : (n, d_high) high-dimensional data.
    X_low   : (n, d_low)  low-dimensional embedding.
    k       : neighbourhood size (not used directly in this implementation
              but kept for API compatibility with CCVGAE).
    verbose : if True, print a summary to stdout.

    Returns
    -------
    dict with keys: ``distance_correlation``, ``Q_local``, ``Q_global``,
    ``K_max``, ``overall_quality``.
    """
    if not isinstance(X_high, np.ndarray) or not isinstance(X_low, np.ndarray):
        raise TypeError("X_high and X_low must be numpy arrays")
    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {X_high.shape[0]} vs {X_low.shape[0]}"
        )
    if X_high.ndim != 2 or X_low.ndim != 2:
        raise ValueError("X_high and X_low must be 2-D arrays")
    if k >= X_high.shape[0]:
        raise ValueError(f"k ({k}) must be < n_samples ({X_high.shape[0]})")

    results: dict = {}

    # 1. Distance correlation (Spearman)
    try:
        D_high = pairwise_distances(X_high)
        D_low = pairwise_distances(X_low)
        r, _ = spearmanr(D_high.ravel(), D_low.ravel())
        results["distance_correlation"] = float(r) if not np.isnan(r) else 0.0
    except Exception as exc:
        warnings.warn(f"distance_correlation error: {exc}")
        results["distance_correlation"] = 0.0
        D_high = pairwise_distances(X_high)
        D_low = pairwise_distances(X_low)

    # 2. Ranking matrices
    rank_high = get_ranking_matrix(D_high)
    rank_low = get_ranking_matrix(D_low)

    # 3. Co-ranking matrix
    corank = get_coranking_matrix(rank_high, rank_low)

    # 4. Q metrics
    qnx_values = compute_qnx_series(corank)
    Q_local, Q_global, K_max = get_q_local_global(qnx_values)

    results["Q_local"] = Q_local
    results["Q_global"] = Q_global
    results["K_max"] = K_max
    results["overall_quality"] = float(
        np.mean([results["distance_correlation"], Q_local, Q_global])
    )

    if verbose:
        print(
            f"[DRE] dist_corr={results['distance_correlation']:.4f}  "
            f"Q_local={Q_local:.4f}  Q_global={Q_global:.4f}  "
            f"K_max={K_max}  overall={results['overall_quality']:.4f}"
        )

    return results
