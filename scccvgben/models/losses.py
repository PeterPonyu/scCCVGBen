"""Loss functions for CCVGAE training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nb_loss(pred_x: torch.Tensor, x: torch.Tensor, dispersion: torch.Tensor) -> torch.Tensor:
    """Negative-binomial log-likelihood loss for scRNA counts.

    pred_x: (N, G) predicted mean (after softmax * library size)
    x:      (N, G) raw counts
    dispersion: (G,) or scalar — log-dispersion parameter (exponentiated internally)
    Returns scalar (mean over cells and genes).
    """
    eps = 1e-8
    theta = torch.exp(dispersion)          # (G,) or scalar
    mu = pred_x
    log_theta_mu = torch.log(theta + mu + eps)
    ll = (
        theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return -ll.sum(-1).mean()


def mse_loss(pred_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error reconstruction loss for scATAC LSI features.

    pred_x: (N, D), x: (N, D). Returns scalar.
    """
    return F.mse_loss(pred_x, x)


def kl_loss(q_m: torch.Tensor, q_s: torch.Tensor) -> torch.Tensor:
    """KL divergence from posterior N(q_m, softplus(q_s)) to N(0, I).

    q_m: (N, latent_dim), q_s: (N, latent_dim) log-std pre-activation.
    Returns scalar (mean over cells, sum over latent dims).
    """
    std = F.softplus(q_s) + 1e-6
    # KL( N(mu, std) || N(0,1) ) = 0.5 * (mu^2 + std^2 - 1 - log(std^2))
    kl = 0.5 * (q_m.pow(2) + std.pow(2) - 1.0 - torch.log(std.pow(2) + 1e-8))
    return kl.sum(-1).mean()


def adj_loss(
    pred_a: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """BCE between predicted dense adjacency logits and ground-truth edges.

    pred_a:     (N, N) logit matrix (output of BilinearDecoder / InnerProductDecoder)
    edge_index: (2, E) ground-truth edge indices
    num_nodes:  N
    Returns scalar BCE.
    """
    target = torch.zeros(num_nodes, num_nodes, device=pred_a.device)
    target[edge_index[0], edge_index[1]] = 1.0
    return F.binary_cross_entropy_with_logits(pred_a, target)
