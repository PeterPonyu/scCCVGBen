"""Test scCCVGBen forward pass: shapes, no NaN, backward pass."""

from __future__ import annotations

import numpy as np
import torch

from scccvgben.models import ScCCVGBenModel
from scccvgben.graphs import build_knn_euclidean

N_CELLS = 100
N_FEATS = 50
LATENT_DIM = 10
I_DIM = 5
HIDDEN = 64


def _build_synthetic_graph():
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((N_CELLS, N_FEATS)).astype(np.float32)
    edge_index, edge_weight = build_knn_euclidean(X_np, k=10)
    x = torch.from_numpy(X_np)
    return x, edge_index, edge_weight


def test_scccvgben_model_forward_shapes():
    """Forward returns a 7-tuple with correct shapes."""
    model = ScCCVGBenModel(
        in_dim=N_FEATS,
        hidden=HIDDEN,
        latent_dim=LATENT_DIM,
        i_dim=I_DIM,
        encoder_name="GAT",
    )
    model.eval()

    x, edge_index, edge_weight = _build_synthetic_graph()

    with torch.no_grad():
        result = model(x, edge_index, edge_weight)

    assert len(result) == 7, f"Expected 7-tuple, got {len(result)}"
    q_z, q_m, q_s, pred_a, pred_x, le, pred_xl = result

    assert q_z.shape == (N_CELLS, LATENT_DIM), f"q_z shape {q_z.shape}"
    assert q_m.shape == (N_CELLS, LATENT_DIM), f"q_m shape {q_m.shape}"
    assert q_s.shape == (N_CELLS, LATENT_DIM), f"q_s shape {q_s.shape}"
    assert pred_a.shape == (N_CELLS, N_CELLS), f"pred_a shape {pred_a.shape}"
    assert pred_x.shape == (N_CELLS, N_FEATS), f"pred_x shape {pred_x.shape}"
    assert le.shape == (N_CELLS, I_DIM), f"le shape {le.shape}"
    assert pred_xl.shape == (N_CELLS, N_FEATS), f"pred_xl shape {pred_xl.shape}"

    for name, t in [("q_z", q_z), ("q_m", q_m), ("q_s", q_s),
                     ("pred_a", pred_a), ("pred_x", pred_x), ("le", le), ("pred_xl", pred_xl)]:
        assert not torch.isnan(t).any(), f"{name} contains NaN"


def test_scccvgben_backward():
    """Loss computation and backward pass should not raise."""
    model = ScCCVGBenModel(
        in_dim=N_FEATS,
        hidden=HIDDEN,
        latent_dim=LATENT_DIM,
        i_dim=I_DIM,
        encoder_name="GAT",
    )
    model.train()

    x, edge_index, edge_weight = _build_synthetic_graph()

    q_z, q_m, q_s, pred_a, pred_x, le, pred_xl = model(x, edge_index, edge_weight)

    # Simple surrogate loss: reconstruction + graph
    recon_loss = torch.nn.functional.mse_loss(pred_x, x)
    adj_target = torch.zeros(N_CELLS, N_CELLS)
    adj_target[edge_index[0], edge_index[1]] = 1.0
    graph_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_a, adj_target)
    loss = recon_loss + graph_loss

    # Should not raise
    loss.backward()
