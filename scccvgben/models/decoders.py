"""Decoder modules for CCVGAE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearDecoder(nn.Module):
    """Reconstruct feature matrix from latent z.

    Maps z (N, latent_dim) -> pred_x (N, out_dim) via MLP.
    Includes a log-dispersion parameter for NB likelihood.
    """

    def __init__(self, latent_dim: int, out_dim: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = latent_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_ch, hidden), nn.ReLU()]
            in_ch = hidden
        layers.append(nn.Linear(in_ch, out_dim))
        layers.append(nn.Softmax(dim=-1))   # normalized means for NB
        self.net = nn.Sequential(*layers)
        self.disp = nn.Parameter(torch.zeros(out_dim))  # log-dispersion

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, latent_dim) -> pred_x: (N, out_dim)."""
        return self.net(z)


class BilinearDecoder(nn.Module):
    """Reconstruct adjacency via sigmoid(z @ W @ z.T).

    pred_a (N, N) — full dense adjacency logits.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(latent_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, latent_dim) -> pred_a: (N, N) logits."""
        return z @ self.W @ z.t()


class InnerProductDecoder(nn.Module):
    """Reconstruct adjacency via sigmoid(z @ z.T).

    pred_a (N, N) — full dense adjacency logits.
    """

    def __init__(self, latent_dim: int):  # latent_dim kept for API consistency
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, latent_dim) -> pred_a: (N, N) logits."""
        return z @ z.t()


class MLPDecoder(nn.Module):
    """Per-edge probability decoder.

    Concatenates z[i] and z[j] for each edge then passes through MLP -> scalar logit.
    """

    def __init__(self, latent_dim: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = latent_dim * 2
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_ch, hidden), nn.ReLU()]
            in_ch = hidden
        layers.append(nn.Linear(in_ch, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """z: (N, latent_dim), edge_index: (2, E) -> logits: (E,)."""
        src, dst = edge_index
        return self.net(torch.cat([z[src], z[dst]], dim=-1)).squeeze(-1)
