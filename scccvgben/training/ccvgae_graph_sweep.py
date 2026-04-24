"""scCCVGBen Axis B: CCVGAE × 5 graph constructions.

scCCVGBen-original extension — NOT in upstream CCVGAE. Compares how graph
construction (the backbone for encoder message passing) affects latent
quality when encoder is held fixed to GAT.

Graph constructions (from scccvgben.graphs.construction):
  1. kNN_euclidean      — default; same as Axis A GAT cell (reused, not re-run)
  2. kNN_cosine         — cosine-similarity kNN
  3. snn                — shared-nearest-neighbour graph
  4. mutual_knn         — mutual kNN (stricter connectivity)
  5. gaussian_threshold — Gaussian-kernel thresholded dense graph

Injection mechanism:
  - Pre-compute X_PCA (CGVAE_env uses adata.obsm['X_PCA'] internally).
  - Build custom graph via scccvgben.graphs.construction.build(method, X_PCA, k).
  - Construct symmetric scipy.sparse CSR and assign to adata.obsp['connectivities'].
  - Monkey-patch sc.pp.neighbors to no-op during CGVAE_env._register_adata so our
    pre-computed graph survives (CGVAE_env unconditionally calls sc.pp.neighbors
    which would otherwise overwrite connectivities).
  - Fit CGVAE_agent with graph_type='GAT' fixed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import anndata as ad
import torch

from scccvgben.external.ccvgae_core.cgvae import CGVAE_agent
from scccvgben.training.metrics import compute_metrics
from scccvgben.training.ccvgae_runner import (
    CCVGAE_DEFAULTS,
    preprocess_scrna_ccvgae,
    _get_labels,
)


# Method naming for Axis B: CCVGAE_GAT_{graph} — so rows distinguish from Axis A
GRAPH_METHODS = [
    "kNN_euclidean",         # shared with Axis A (optional skip)
    "kNN_cosine",
    "snn",
    "mutual_knn",
    "gaussian_threshold",
]


def _build_obsp_connectivities(adata: ad.AnnData, graph_method: str, k: int = 15) -> None:
    """In-place: compute graph on adata.obsm['X_PCA'] and store as obsp.

    Uses scccvgben.graphs.construction.build to produce (edge_index, edge_weight),
    then forms a symmetric scipy.sparse CSR connectivity matrix in obsp.
    """
    from scccvgben.graphs.construction import build
    if "X_PCA" not in adata.obsm:
        # CGVAE_env uses uppercase 'X_PCA' — pre-compute with scanpy (lower-case pca key)
        sc.pp.pca(adata, n_comps=min(50, max(2, adata.n_vars - 1)))
        adata.obsm["X_PCA"] = adata.obsm["X_pca"]

    X_emb = np.asarray(adata.obsm["X_PCA"])
    edge_index, edge_weight = build(graph_method, X_emb, k=k)
    # edge_index: (2, E); edge_weight: (E,); both torch tensors
    ei = edge_index.detach().cpu().numpy()
    ew = edge_weight.detach().cpu().numpy()
    n = adata.n_obs
    conn = sp.csr_matrix(
        (ew.astype(np.float32), (ei[0].astype(np.int64), ei[1].astype(np.int64))),
        shape=(n, n),
    )
    # Symmetrise + strip self-loops (scanpy convention)
    conn = conn.maximum(conn.T)
    conn.setdiag(0.0)
    conn.eliminate_zeros()
    adata.obsp["connectivities"] = conn
    # Minimal stub so CGVAE_env reads it without KeyError
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "connectivities",
        "params": {"method": f"scccvgben:{graph_method}", "n_neighbors": int(k)},
    }


class _NoOpNeighbors:
    """Context manager: temporarily replace sc.pp.neighbors with a no-op
    so CGVAE_env's _register_adata doesn't overwrite our pre-computed graph.
    """
    def __enter__(self):
        self._orig = sc.pp.neighbors
        def _noop(adata, **kw):  # type: ignore[no-redef]
            # Assumes caller has already populated adata.obsp['connectivities']
            return None
        sc.pp.neighbors = _noop
        return self

    def __exit__(self, *exc):
        sc.pp.neighbors = self._orig
        return False


def run_ccvgae_graph_one(
    h5ad_path: str | Path,
    graph_method: str = "kNN_euclidean",
    method_name: str | None = None,
    data_type: str = "trajectory",
    k: int = 15,
    silent: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """Run CCVGAE with encoder=GAT fixed + custom graph construction."""
    cfg = {**CCVGAE_DEFAULTS, **overrides}
    cfg["graph_type"] = "GAT"  # fixed for Axis B

    adata = ad.read_h5ad(str(h5ad_path))
    labels = _get_labels(adata)
    adata = preprocess_scrna_ccvgae(
        adata,
        subsample_cells=cfg.pop("subsample_cells", 3000),
        n_top_genes=cfg.pop("n_top_genes", 2000),
    )

    epochs = int(cfg.pop("epochs", 100))
    device = cfg.pop("device", None) or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    cfg.pop("tech", None)
    cfg.pop("encoder_type", None)

    if graph_method == "kNN_euclidean":
        # Use CCVGAE's default scanpy path (same as Axis A). No injection.
        agent = CGVAE_agent(
            adata=adata, layer="counts", **cfg, device=device,
        ).fit(epochs=epochs, silent=silent)
    else:
        # Inject custom graph into obsp; monkey-patch sc.pp.neighbors so the
        # internal CGVAE_env call doesn't overwrite it.
        _build_obsp_connectivities(adata, graph_method, k=k)
        with _NoOpNeighbors():
            agent = CGVAE_agent(
                adata=adata, layer="counts", **cfg, device=device,
            ).fit(epochs=epochs, silent=silent)

    latent = agent.get_latent()
    if labels is not None and len(labels) != latent.shape[0]:
        labels = None

    metrics_df = compute_metrics(
        latent,
        X_orig=None,
        labels=labels,
        method_name=method_name or f"scCCVGBen_GAT_{graph_method}",
        data_type=data_type,
    )
    return metrics_df.iloc[0].to_dict()
