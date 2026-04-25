"""Training loop for scCCVGBen."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from ..models.scccvgben_model import ScCCVGBenModel
from ..models.losses import mse_loss, kl_loss, adj_loss
from .configs import LOCKED_CONFIG
from .metrics import compute_metrics


def fit_one(
    model: ScCCVGBenModel,
    data: Data,
    modality: str,
    **cfg,
) -> dict[str, float]:
    """Train scCCVGBen for one dataset and return a metrics dict.

    Parameters
    ----------
    model    : ScCCVGBenModel instance (freshly initialised)
    data     : torch_geometric Data with .x, .edge_index, .edge_attr (optional), .y (optional)
    modality : 'scrna' or 'scatac'
    **cfg    : override any key from LOCKED_CONFIG[modality]

    Returns dict with all compute_metrics columns as floats.
    """
    base = LOCKED_CONFIG[modality].copy()
    base.update(cfg)

    seed: int = base.get("seed", 42)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_attr.to(device) if data.edge_attr is not None else None
    labels = data.y.cpu().numpy() if data.y is not None else None

    epochs: int = base.get("epochs", 100)
    lr: float = base.get("lr", 1e-4)
    wd: float = base.get("weight_decay", 1e-5)
    lw: dict = base.get("loss_weights", {"recon": 1.0, "irecon": 0.5, "kl": 0.01, "adj": 1.0})
    # NOTE: scRNA recon is MSE on PCA features (not NB on raw counts) because the model
    # trains on PCA-embedded inputs. Switching to gene-space NB requires carrying raw counts
    # in `data.raw_x` — out of scope for v0.1.
    use_mse: bool = base.get("use_mse_likelihood", True)

    w_recon = lw.get("recon", 1.0)
    w_irecon = lw.get("irecon", 0.5)
    w_kl = lw.get("kl", 0.01)
    w_adj = lw.get("adj", 1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        q_z, q_m, q_s, pred_a, pred_x, le, pred_xl = model(x, edge_index, edge_weight)

        num_nodes = x.size(0)

        if modality == "scrna":
            r_loss = mse_loss(pred_x, x)
            ir_loss = mse_loss(pred_xl, x)
        else:
            r_loss = mse_loss(pred_x, x)
            ir_loss = mse_loss(pred_xl, x)

        kl = kl_loss(q_m, q_s)
        al = adj_loss(pred_a, edge_index, num_nodes)

        loss = w_recon * r_loss + w_irecon * ir_loss + w_kl * kl + w_adj * al
        loss.backward()
        optimizer.step()

    # ── extract latent and compute metrics ────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, q_m_final, _, _, _, _, _ = model(x, edge_index, edge_weight)

    Z = q_m_final.cpu().numpy()
    X_orig = x.cpu().numpy()
    metrics_df = compute_metrics(Z, X_orig, labels=labels, method_name="scCCVGBen")
    return metrics_df.iloc[0].to_dict()
