"""Canonical CCVGAE model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .encoder_registry import ENCODER_REGISTRY, build_encoder
from .decoders import LinearDecoder, BilinearDecoder, InnerProductDecoder


class _GraphEncoderStack(nn.Module):
    """Multi-layer GNN encoder with residual connection, producing mean + log-std heads."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        latent_dim: int,
        encoder_name: str,
        n_layers: int,
        dropout: float,
        residual: bool,
        **encoder_kwargs,
    ):
        super().__init__()
        self.residual = residual

        # Build hidden layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.drops = nn.ModuleList()

        in_ch = in_dim
        for _ in range(n_layers):
            self.convs.append(build_encoder(encoder_name, in_ch, hidden, **encoder_kwargs))
            self.bns.append(nn.BatchNorm1d(hidden))
            self.drops.append(nn.Dropout(dropout))
            in_ch = hidden

        # Mean and log-std heads (same conv type)
        self.conv_mean = build_encoder(encoder_name, hidden, latent_dim, **encoder_kwargs)
        self.conv_logstd = build_encoder(encoder_name, hidden, latent_dim, **encoder_kwargs)

        self.relu = nn.ReLU()
        self._needs_edge_attr = ENCODER_REGISTRY[encoder_name]["needs_edge_attr"]

    def _conv_forward(
        self, conv, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None
    ) -> torch.Tensor:
        if self._needs_edge_attr and edge_weight is not None:
            return conv(x, edge_index, edge_attr=edge_weight)
        # For encoders that don't consume edge attributes (GATv2, TransformerConv,
        # SuperGAT, etc.), passing a non-None edge_weight positionally triggers
        # their internal edge_update path which asserts lin_edge is not None —
        # a layer we never built. Drop edge_weight explicitly.
        try:
            return conv(x, edge_index)
        except TypeError:
            return conv(x, edge_index, edge_weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (q_z, q_m, q_s). q_z: (N, latent_dim) sampled via reparameterisation."""
        residual = None
        for i, (conv, bn, drop) in enumerate(zip(self.convs, self.bns, self.drops)):
            x = self._conv_forward(conv, x, edge_index, edge_weight)
            x = bn(x)
            x = self.relu(x)
            x = drop(x)
            if self.residual and i == 0:
                residual = x
        if self.residual and residual is not None:
            x = x + residual

        q_m = self._conv_forward(self.conv_mean, x, edge_index, edge_weight)
        q_s = self._conv_forward(self.conv_logstd, x, edge_index, edge_weight)

        std = F.softplus(q_s) + 1e-6
        q_z = Normal(q_m, std).rsample()
        return q_z, q_m, q_s


class CCVGAE(nn.Module):
    """Coupled-Centroid Variational Graph Autoencoder.

    Attributes (shapes given for N cells, F features, L latent_dim, I i_dim):
        encoder:         _GraphEncoderStack  -> q_z (N, L)
        latent_encoder:  Linear(L, I)        -> le  (N, I)   projection bottleneck
        latent_decoder:  Linear(I, L)        -> ld  (N, L)   back to latent space
        feature_decoder: LinearDecoder       -> pred_x / pred_xl (N, F)
        graph_decoder:   Bilinear|IP decoder -> pred_a (N, N) logits
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        latent_dim: int = 10,
        i_dim: int = 5,
        encoder_name: str = "GAT",
        graph_decoder: str = "bilinear",
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
        **encoder_kwargs,
    ):
        super().__init__()

        self.encoder = _GraphEncoderStack(
            in_dim=in_dim,
            hidden=hidden,
            latent_dim=latent_dim,
            encoder_name=encoder_name,
            n_layers=n_enc_layers,
            dropout=dropout,
            residual=residual,
            **encoder_kwargs,
        )

        # Projection bottleneck — the "centroid coupling"
        self.latent_encoder = nn.Linear(latent_dim, i_dim)   # (N,L) -> (N,I)
        self.latent_decoder = nn.Linear(i_dim, latent_dim)   # (N,I) -> (N,L)

        self.feature_decoder = LinearDecoder(
            latent_dim=latent_dim,
            out_dim=in_dim,
            hidden=hidden,
            n_layers=n_dec_layers,
        )

        _gd = graph_decoder.lower()
        if _gd == "bilinear":
            self.graph_decoder: nn.Module = BilinearDecoder(latent_dim)
        elif _gd in ("inner_product", "innerproduct", "ip"):
            self.graph_decoder = InnerProductDecoder(latent_dim)
        else:
            raise ValueError(f"Unknown graph_decoder '{graph_decoder}'. Use 'bilinear' or 'inner_product'.")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            q_z:     (N, L) — sampled latent
            q_m:     (N, L) — posterior mean
            q_s:     (N, L) — posterior log-std (pre-softplus)
            pred_a:  (N, N) — adjacency logits
            pred_x:  (N, F) — feature reconstruction from q_z
            le:      (N, I) — bottleneck (latent_encoder output)
            pred_xl: (N, F) — feature reconstruction from latent_decoder path
        """
        q_z, q_m, q_s = self.encoder(x, edge_index, edge_weight)

        le = self.latent_encoder(q_z)   # (N, I)
        ld = self.latent_decoder(le)    # (N, L)

        pred_a = self.graph_decoder(q_z)    # (N, N)
        pred_x = self.feature_decoder(q_z)  # (N, F)
        pred_xl = self.feature_decoder(ld)  # (N, F)

        return q_z, q_m, q_s, pred_a, pred_x, le, pred_xl
