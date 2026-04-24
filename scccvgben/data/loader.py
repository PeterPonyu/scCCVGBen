"""Load h5ad files into torch_geometric Data objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import anndata as ad
from torch_geometric.data import Data

from .preprocessing import preprocess_scrna, preprocess_scatac
from .labels import get_labels
from ..graphs.construction import build


def load_dataset(
    h5ad_path: str | Path,
    modality: str,
    k: int = 15,
    graph_method: str = "kNN_euclidean",
    **cfg,
) -> Data:
    """Read h5ad, preprocess, build graph, return torch_geometric Data.

    Parameters
    ----------
    h5ad_path    : path to .h5ad file
    modality     : 'scrna' or 'scatac'
    k            : number of nearest neighbours for graph construction
    graph_method : one of the keys in graphs.construction._BUILDERS
    **cfg        : forwarded to preprocess_* (n_top_genes/peaks, n_pcs/lsi, subsample)

    Returns Data with:
        .x           (N, D) float32 — PCA or LSI embedding
        .edge_index  (2, E) int64
        .edge_attr   (E,)   float32
        .y           (N,)   int64 label codes, or None
    """
    adata = ad.read_h5ad(h5ad_path)

    if modality == "scrna":
        n_top_genes = cfg.get("hvg_count", cfg.get("n_top_genes", 2000))
        n_pcs = cfg.get("pca_dim", cfg.get("n_pcs", 50))
        subsample = cfg.get("subsample_cells", cfg.get("subsample", None))
        adata = preprocess_scrna(adata, n_top_genes=n_top_genes, n_pcs=n_pcs, subsample=subsample)
        X_embed = adata.obsm["X_pca"].astype(np.float32)
    elif modality == "scatac":
        n_top_peaks = cfg.get("hv_peak_count", cfg.get("n_top_peaks", 2000))
        n_lsi = cfg.get("lsi_dim", cfg.get("n_lsi", 50))
        subsample = cfg.get("subsample_cells", cfg.get("subsample", None))
        adata = preprocess_scatac(adata, n_top_peaks=n_top_peaks, n_lsi=n_lsi, subsample=subsample)
        X_embed = adata.obsm["X_lsi"].astype(np.float32)
    else:
        raise ValueError(f"Unknown modality '{modality}'. Use 'scrna' or 'scatac'.")

    edge_index, edge_weight = build(graph_method, X_embed, k=k)

    x_tensor = torch.from_numpy(X_embed)

    # Labels: unified via scccvgben.data.labels.get_labels, stored as int64 codes.
    import pandas as pd
    y_tensor = None
    labels_str = get_labels(adata)
    if labels_str is not None:
        y_tensor = torch.tensor(
            pd.Categorical(labels_str).codes, dtype=torch.long
        )

    return Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y_tensor,
    )
