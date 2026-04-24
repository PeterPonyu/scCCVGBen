"""Registry of supported graph encoder layers and builder helper."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    TransformerConv,
    SuperGATConv,
    GCNConv,
    SAGEConv,
    GINConv,
    ChebConv,
    EdgeConv,
    ARMAConv,
    SGConv,
    TAGConv,
)

# PNAConv optional — may not be in all pyg versions
try:
    from torch_geometric.nn import PNAConv as _PNAConv
    _HAS_PNA = True
except ImportError:
    _HAS_PNA = False

ENCODER_REGISTRY: dict[str, dict] = {
    # ── attention family ─────────────────────────────────────────────────────
    "GAT": {
        "class": GATConv,
        "family": "attention",
        "init_kwargs": {"heads": 4, "concat": False, "dropout": 0.1},
        "needs_edge_attr": False,
    },
    "GATv2": {
        "class": GATv2Conv,
        "family": "attention",
        "init_kwargs": {"heads": 4, "concat": False, "dropout": 0.1},
        "needs_edge_attr": False,
    },
    "TransformerConv": {
        "class": TransformerConv,
        "family": "attention",
        "init_kwargs": {"heads": 4, "concat": False, "dropout": 0.1},
        "needs_edge_attr": False,
    },
    "SuperGAT": {
        "class": SuperGATConv,
        "family": "attention",
        "init_kwargs": {"heads": 4, "concat": False, "dropout": 0.1},
        "needs_edge_attr": False,
    },
    # ── message-passing family ───────────────────────────────────────────────
    "GCN": {
        "class": GCNConv,
        "family": "message-passing",
        "init_kwargs": {},
        "needs_edge_attr": False,
    },
    "GraphSAGE": {
        "class": SAGEConv,
        "family": "message-passing",
        "init_kwargs": {},
        "needs_edge_attr": False,
    },
    "GIN": {
        "class": GINConv,
        "family": "message-passing",
        "init_kwargs": {},          # inner MLP built at call time
        "needs_edge_attr": False,
    },
    "ChebNet": {
        "class": ChebConv,
        "family": "message-passing",
        "init_kwargs": {"K": 2},
        "needs_edge_attr": False,
    },
    "EdgeConv": {
        "class": EdgeConv,
        "family": "message-passing",
        "init_kwargs": {},          # nn callable built at call time
        "needs_edge_attr": False,
    },
    "ARMAConv": {
        "class": ARMAConv,
        "family": "message-passing",
        "init_kwargs": {},
        "needs_edge_attr": False,
    },
    "SGConv": {
        "class": SGConv,
        "family": "message-passing",
        "init_kwargs": {"K": 2},
        "needs_edge_attr": False,
    },
    "TAGConv": {
        "class": TAGConv,
        "family": "message-passing",
        "init_kwargs": {"K": 3},
        "needs_edge_attr": False,
    },
    "PNAConv": {
        "class": _PNAConv if _HAS_PNA else None,
        "family": "message-passing",
        "init_kwargs": {
            "aggregators": ["mean", "min", "max", "std"],
            "scalers": ["identity", "amplification", "attenuation"],
        },
        "needs_edge_attr": False,
    },
}


def build_encoder(
    name: str,
    in_dim: int,
    out_dim: int,
    deg: torch.Tensor | None = None,
    **extra_kwargs,
) -> nn.Module:
    """Instantiate a single graph conv layer from the registry.

    Handles per-class quirks:
    - GIN: wraps an inner Linear MLP
    - EdgeConv: wraps an inner two-layer MLP
    - PNAConv: requires deg histogram (falls back to ones if not provided)
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(ENCODER_REGISTRY)}")

    entry = ENCODER_REGISTRY[name]
    cls = entry["class"]
    if cls is None:
        raise ImportError(f"Encoder '{name}' requires a package that is not installed.")

    kwargs = {**entry["init_kwargs"], **extra_kwargs}

    if name == "GIN":
        inner = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        return cls(nn=inner, **kwargs)

    if name == "EdgeConv":
        inner = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        return cls(nn=inner, **kwargs)

    if name == "PNAConv":
        if deg is None:
            deg = torch.ones(10, dtype=torch.long)
        return cls(in_channels=in_dim, out_channels=out_dim, deg=deg, **kwargs)

    return cls(in_channels=in_dim, out_channels=out_dim, **kwargs)
