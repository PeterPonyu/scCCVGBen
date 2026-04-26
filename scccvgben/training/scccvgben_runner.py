"""scCCVGBen one-dataset runner aligned with the reference benchmark routine.

Flow per dataset (mirrors the reference run_single path exactly):
  1. Load h5ad.
  2. Preprocess: subsample(3000) -> normalize_total(1e4) -> log1p -> HVG(2000) -> subset.
     Raw counts preserved in adata.layers['counts'] for CGVAE_agent.
  3. Instantiate CGVAE_agent(adata, **cfg) -> .fit(epochs) -> .get_latent()
  4. compute_metrics(latent, labels, data_type).

All configs default to reference benchmark values (lr=1e-4, epochs=100,
hidden_dim=128, hidden_layers=2, latent_dim=10, i_dim=5, w_*=1.0,
subgraph_size=300). graph_type defaults to 'GAT' for Axis A/B primary cell.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
import anndata as ad

from scccvgben.external.reference_core.cgvae import CGVAE_agent
from scccvgben.training.metrics import compute_metrics


SCCCVGBEN_DEFAULTS: dict[str, Any] = {
    # Preprocess
    "subsample_cells": 3000,
    "n_top_genes": 2000,
    # CGVAE_agent config (matching reference benchmark defaults)
    "hidden_dim": 128,
    "hidden_layers": 2,
    "latent_dim": 10,
    "i_dim": 5,
    "lr": 1e-4,
    "w_recon": 1.0,
    "w_irecon": 1.0,
    "w_kl": 1.0,
    "w_adj": 1.0,
    "subgraph_size": 300,
    "num_subgraphs_per_epoch": 10,
    "epochs": 100,
    "tech": "PCA",
    "n_neighbors": 15,
    "graph_type": "GAT",
    "encoder_type": "graph",
    "device": None,  # auto pick cuda/cpu
}


def _get_labels(adata: ad.AnnData) -> np.ndarray | None:
    """Deprecated label-extraction helper (returns ``None``).

    The active metric protocol is fully self-supervised: KMeans on the
    pre-processed input X versus KMeans on the latent (see
    ``scccvgben.training.metrics`` docstring). No ``adata.obs`` column is
    consulted for clustering metrics. Kept as a stub so downstream callers
    that still pass a ``labels`` keyword keep working.
    """
    return None


def preprocess_scrna_scccvgben(adata: ad.AnnData,
                            subsample_cells: int = 3000,
                            n_top_genes: int = 2000,
                            random_state: int = 42,
                            min_counts_per_cell: int = 200,
                            min_cells_per_gene: int = 10) -> ad.AnnData:
    """Preprocess matching the reference benchmark hyperparameter routine.

    QC filter -> subsample -> normalize_total(1e4) -> log1p -> HVG -> subset.
    Raw counts stashed in adata.layers['counts'] for CGVAE_agent's
    layer='counts' parameter.

    QC filter (added 2026-04-25):
        Drops empty droplets (cell sum < ``min_counts_per_cell``) and
        zero-variance genes (expressed in fewer than ``min_cells_per_gene``
        cells) BEFORE the random subsample. Without this step, raw cellranger
        matrices like GSE128033_new (737k barcodes, 82% empty droplets)
        produce a near-singular feature matrix after subsample, raising
        "near singularities" / "reciprocal condition number" inside the
        downstream KMeans / SVD path. The thresholds match standard scanpy
        defaults for raw scRNA matrices.
    """
    # QC: remove empty cells and zero-variance genes before subsampling
    if min_counts_per_cell > 0:
        sc.pp.filter_cells(adata, min_counts=min_counts_per_cell)
    if min_cells_per_gene > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    # Subsample to cap training time
    if adata.n_obs > subsample_cells:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, size=subsample_cells, replace=False)
        adata = adata[np.sort(idx)].copy()

    # Preserve raw counts for CGVAE_agent(layer='counts')
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_top = min(n_top_genes, max(2, adata.n_vars - 1))
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat_v3",
                                layer="counts")
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


def run_scccvgben_one(
    h5ad_path: str | Path,
    graph_type: str = "GAT",
    method_name: str | None = None,
    data_type: str = "trajectory",
    silent: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """Run scCCVGBen on one h5ad and return a metrics dict (27 cols).

    Parameters
    ----------
    h5ad_path   : path to input h5ad (raw or log+HVG preprocessed)
    graph_type  : encoder family ('GAT', 'GATv2', 'GCN', 'GraphSAGE', ...)
    method_name : value for the 'method' column (defaults to f'scCCVGBen_{graph_type}')
    data_type   : 'trajectory' or 'steady_state' for LSE scoring
    silent      : quiet tqdm during .fit()
    **overrides : override any SCCCVGBEN_DEFAULTS key
    """
    import torch

    cfg = {**SCCCVGBEN_DEFAULTS, **overrides}
    cfg["graph_type"] = graph_type

    adata = ad.read_h5ad(str(h5ad_path))
    labels = _get_labels(adata)

    adata = preprocess_scrna_scccvgben(
        adata,
        subsample_cells=cfg.pop("subsample_cells", 3000),
        n_top_genes=cfg.pop("n_top_genes", 2000),
    )

    epochs = int(cfg.pop("epochs", 100))
    device = cfg.pop("device", None) or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Drop benchmark-only keys before passing to CGVAE_agent
    cfg.pop("tech", None)  # already default in CGVAE_agent
    cfg.pop("encoder_type", None)

    agent = CGVAE_agent(
        adata=adata,
        layer="counts",
        **cfg,
        device=device,
    ).fit(epochs=epochs, silent=silent)
    latent = agent.get_latent()

    if labels is not None:
        n_labels = len(labels)
        if n_labels != latent.shape[0]:
            labels = None  # subsampled — labels may not align

    metrics_df = compute_metrics(
        latent,
        X_orig=None,
        labels=labels,
        method_name=method_name or f"scCCVGBen_{graph_type}",
        data_type=data_type,
    )
    return metrics_df.iloc[0].to_dict()
