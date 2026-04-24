"""Test every encoder in ENCODER_REGISTRY: instantiate, forward, shape, no NaN."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from scccvgben.models import ENCODER_REGISTRY, build_encoder

N_NODES = 100
IN_DIM = 50
OUT_DIM = 128

# Build a synthetic random kNN-style edge_index for 100 nodes
def _synthetic_edge_index(n: int = N_NODES, k: int = 10) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(0)
    sources = torch.arange(n).repeat_interleave(k)
    targets = torch.randint(0, n, (n * k,), generator=rng)
    # Add self-loops
    sl = torch.arange(n)
    edge_index = torch.stack([
        torch.cat([sources, sl]),
        torch.cat([targets, sl]),
    ], dim=0)
    return edge_index


_EDGE_INDEX = _synthetic_edge_index()
_X = torch.randn(N_NODES, IN_DIM)


@pytest.mark.parametrize("name", [k for k, v in ENCODER_REGISTRY.items() if v["class"] is not None])
def test_encoder_forward_shape_no_nan(name: str):
    """Instantiate encoder, run forward, check output shape and no NaN."""
    enc = build_encoder(name, in_dim=IN_DIM, out_dim=OUT_DIM)
    enc.eval()
    with torch.no_grad():
        out = enc(_X, _EDGE_INDEX)
    assert out.shape == (N_NODES, OUT_DIM), (
        f"{name}: expected ({N_NODES}, {OUT_DIM}), got {out.shape}"
    )
    assert not torch.isnan(out).any(), f"{name}: output contains NaN"
