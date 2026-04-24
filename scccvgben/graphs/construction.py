"""Graph construction functions for single-cell data.

All functions return:
    edge_index : torch.Tensor, shape (2, E), dtype int64
    edge_weight: torch.Tensor, shape (E,),   dtype float32
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph


def _to_numpy(X) -> np.ndarray:
    """Accept either np.ndarray or torch.Tensor; return np.ndarray."""
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    return np.asarray(X, dtype=np.float32)


# ── helpers ──────────────────────────────────────────────────────────────────

def _scipy_to_tensors(
    A,
    symmetrize: bool = True,
    self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a scipy sparse kNN graph to (edge_index, edge_weight)."""
    if symmetrize:
        A = (A + A.T)
        A.data[:] = 1.0
    A = A.tocoo()
    rows, cols = A.row.astype(np.int64), A.col.astype(np.int64)
    weights = A.data.astype(np.float32)
    if self_loops:
        n = A.shape[0]
        sl = np.arange(n, dtype=np.int64)
        rows = np.concatenate([rows, sl])
        cols = np.concatenate([cols, sl])
        weights = np.concatenate([weights, np.ones(n, dtype=np.float32)])
    edge_index = torch.from_numpy(np.stack([rows, cols])).long()
    edge_weight = torch.from_numpy(weights).float()
    return edge_index, edge_weight


# ── public builders ───────────────────────────────────────────────────────────

def build_knn_euclidean(
    X,
    k: int = 15,
    symmetrize: bool = True,
    self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k-NN graph with Euclidean distance.

    Accepts np.ndarray or torch.Tensor.
    edge_index: (2, E) int64  |  edge_weight: (E,) float32
    """
    X = _to_numpy(X)
    A = kneighbors_graph(X, n_neighbors=k, metric="euclidean", mode="connectivity", include_self=False)
    return _scipy_to_tensors(A, symmetrize=symmetrize, self_loops=self_loops)


def build_knn_cosine(
    X,
    k: int = 15,
    symmetrize: bool = True,
    self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k-NN graph with cosine distance.

    Accepts np.ndarray or torch.Tensor.
    edge_index: (2, E) int64  |  edge_weight: (E,) float32
    """
    X = _to_numpy(X)
    A = kneighbors_graph(X, n_neighbors=k, metric="cosine", mode="connectivity", include_self=False)
    return _scipy_to_tensors(A, symmetrize=symmetrize, self_loops=self_loops)


def build_snn(
    X,
    k: int = 15,
    prune: float = 1 / 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared Nearest Neighbor graph.

    Accepts np.ndarray or torch.Tensor.
    Two nodes share an edge if |kNN(i) ∩ kNN(j)| / k >= prune.
    edge_index: (2, E) int64  |  edge_weight: (E,) float32 (SNN similarity)
    """
    X = _to_numpy(X)
    A = kneighbors_graph(X, n_neighbors=k, metric="euclidean", mode="connectivity", include_self=False)
    A_bool = (A > 0).astype(np.float32)
    A_dense = A_bool.toarray()
    # SNN similarity = |kNN(i) ∩ kNN(j)| via dot product
    snn = A_dense @ A_dense.T  # (N, N) — shared neighbor counts
    snn_norm = snn / k
    mask = snn_norm >= prune
    np.fill_diagonal(mask, False)
    rows, cols = np.where(mask)
    weights = snn_norm[rows, cols].astype(np.float32)
    edge_index = torch.from_numpy(np.stack([rows.astype(np.int64), cols.astype(np.int64)])).long()
    edge_weight = torch.from_numpy(weights).float()
    return edge_index, edge_weight


def build_mutual_knn(
    X,
    k: int = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutual k-NN graph: edge (i,j) only if j ∈ kNN(i) AND i ∈ kNN(j).

    Accepts np.ndarray or torch.Tensor.
    edge_index: (2, E) int64  |  edge_weight: (E,) float32
    """
    X = _to_numpy(X)
    A = kneighbors_graph(X, n_neighbors=k, metric="euclidean", mode="connectivity", include_self=False)
    A_bool = A.astype(bool)
    mutual = A_bool.multiply(A_bool.T)
    mutual = mutual.tocoo()
    rows = mutual.row.astype(np.int64)
    cols = mutual.col.astype(np.int64)
    weights = np.ones(len(rows), dtype=np.float32)
    edge_index = torch.from_numpy(np.stack([rows, cols])).long()
    edge_weight = torch.from_numpy(weights).float()
    return edge_index, edge_weight


def build_gaussian_threshold(
    X,
    sigma: float | None = None,
    threshold: float | None = None,
    **_ignored,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian heat-kernel graph: w_ij = exp(-||xi-xj||^2 / (2*sigma^2)).

    Accepts np.ndarray or torch.Tensor.
    Edges kept where w_ij >= threshold.
    sigma defaults to median pairwise distance; threshold defaults to 0.5.
    `k` and other kwargs are ignored for API uniformity with kNN builders.
    edge_index: (2, E) int64  |  edge_weight: (E,) float32
    """
    from sklearn.metrics import pairwise_distances

    X = _to_numpy(X)
    D = pairwise_distances(X, metric="euclidean").astype(np.float32)
    if sigma is None:
        positive = D[D > 0]
        sigma = float(np.median(positive)) if len(positive) else 1.0
    if sigma <= 0:
        sigma = 1.0
    if threshold is None:
        # threshold=0.5 can yield >50% density on 3000-cell subsample,
        # blowing up GAT attention memory. 0.9 keeps only edges at
        # d < 0.46*sigma, producing a moderate-density sparse graph.
        threshold = 0.9

    W = np.exp(-(D ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    mask = W >= threshold
    rows, cols = np.where(mask)
    if len(rows) == 0 and X.shape[0] > 1:
        nearest = D.copy()
        np.fill_diagonal(nearest, np.inf)
        rows = np.arange(X.shape[0], dtype=np.int64)
        cols = np.argmin(nearest, axis=1).astype(np.int64)
    weights = W[rows, cols].astype(np.float32)
    edge_index = torch.from_numpy(np.stack([rows.astype(np.int64), cols.astype(np.int64)])).long()
    edge_weight = torch.from_numpy(weights).float()
    return edge_index, edge_weight


# ── dispatcher ────────────────────────────────────────────────────────────────

_BUILDERS = {
    "kNN_euclidean": build_knn_euclidean,
    "knn_euclidean": build_knn_euclidean,
    "kNN_cosine": build_knn_cosine,
    "knn_cosine": build_knn_cosine,
    "snn": build_snn,
    "SNN": build_snn,
    "mutual_knn": build_mutual_knn,
    "mutual_kNN": build_mutual_knn,
    "gaussian": build_gaussian_threshold,
    "gaussian_threshold": build_gaussian_threshold,
}


def build(name: str, X, **kw) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to a named graph builder.

    name: one of 'kNN_euclidean', 'kNN_cosine', 'snn', 'mutual_knn', 'gaussian'.
    Accepts np.ndarray or torch.Tensor for X.
    Returns (edge_index: (2,E) int64, edge_weight: (E,) float32).
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if name not in _BUILDERS:
        raise ValueError(f"Unknown graph method '{name}'. Available: {list(_BUILDERS)}")
    return _BUILDERS[name](X, **kw)
