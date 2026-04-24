"""Test all 5 graph construction builders on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scccvgben.graphs import (
    build_knn_euclidean,
    build_knn_cosine,
    build_snn,
    build_mutual_knn,
    build_gaussian_threshold,
)

N_NODES = 50
N_FEATS = 20

rng = np.random.default_rng(123)
_X = rng.standard_normal((N_NODES, N_FEATS)).astype(np.float32)

_BUILDERS = [
    ("kNN_euclidean", build_knn_euclidean),
    ("kNN_cosine", build_knn_cosine),
    ("snn", build_snn),
    ("mutual_knn", build_mutual_knn),
    ("gaussian_threshold", build_gaussian_threshold),
]


@pytest.mark.parametrize("name,builder", _BUILDERS, ids=[b[0] for b in _BUILDERS])
def test_graph_builder_output(name: str, builder):
    """edge_index is (2, E) int64 with valid indices; edge_weight is (E,) float32 or None."""
    edge_index, edge_weight = builder(_X)

    assert isinstance(edge_index, torch.Tensor), f"{name}: edge_index not a Tensor"
    assert edge_index.dtype == torch.int64, f"{name}: edge_index dtype {edge_index.dtype} != int64"
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2, (
        f"{name}: edge_index shape {edge_index.shape}, expected (2, E)"
    )
    E = edge_index.shape[1]
    assert E > 0, f"{name}: no edges returned"
    assert int(edge_index.min()) >= 0, f"{name}: negative node index"
    assert int(edge_index.max()) < N_NODES, f"{name}: node index >= N_NODES"

    if edge_weight is not None:
        assert isinstance(edge_weight, torch.Tensor), f"{name}: edge_weight not a Tensor"
        assert edge_weight.dtype == torch.float32, f"{name}: edge_weight dtype {edge_weight.dtype}"
        assert edge_weight.shape == (E,), (
            f"{name}: edge_weight shape {edge_weight.shape}, expected ({E},)"
        )
