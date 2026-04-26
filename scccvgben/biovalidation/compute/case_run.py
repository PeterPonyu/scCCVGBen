"""End-to-end per-case runner.

Given a :class:`CaseSpec`, train (or load latent for) the case dataset and
produce the full payload dict consumed by ``compose/case_figure.py``.

Returned payload schema (each key is optional — composer renders a
placeholder when missing)::

    {
      "case":              CaseSpec,
      "n_obs": int, "n_vars_post_hvg": int,
      "umap":              ndarray (N, 2),       cell × 2 UMAP
      "latent":            ndarray (N, L),       encoder output
      "condition":         pandas Series (N,) categorical
      "cell_type":         pandas Series (N,) categorical
      "pseudotime":        ndarray (N,) | None,
      "latent_corr":       ndarray (L, L)        absolute Pearson
      "top_k_genes_df":    DataFrame  long-form (dim, rank, gene, rho)
      "expression":        DataFrame (N, K_unique) for the genes referenced
                           in top_k_genes_df only — keeps payload small.
    }
"""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from scccvgben.biovalidation.case_definition import CaseSpec
from scccvgben.training.scccvgben_runner import preprocess_scrna_scccvgben

from .latent_gene_corr import top_k_genes_per_dim, latent_self_correlation

log = logging.getLogger(__name__)


def _ensure_labels(adata: ad.AnnData, case: CaseSpec) -> tuple[pd.Series, pd.Series]:
    """Return (condition, cell_type) Series, computing leiden fall-backs as needed."""
    if case.condition_obs and case.condition_obs in adata.obs.columns:
        cond = adata.obs[case.condition_obs].astype(str)
    else:
        # leiden on PCA → condition fallback (coarse)
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata, n_comps=min(20, adata.n_vars - 1))
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)
        sc.tl.leiden(adata, resolution=0.4, key_added="_cond_leiden", flavor="igraph",
                     directed=False, n_iterations=2)
        cond = adata.obs["_cond_leiden"].astype(str).rename("condition_inferred")

    if case.cell_type_obs and case.cell_type_obs in adata.obs.columns:
        ct = adata.obs[case.cell_type_obs].astype(str)
    else:
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata, n_comps=min(20, adata.n_vars - 1))
        sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)
        sc.tl.leiden(adata, resolution=0.8, key_added="_ct_leiden", flavor="igraph",
                     directed=False, n_iterations=2)
        ct = adata.obs["_ct_leiden"].astype(str).rename("cell_type_inferred")

    return cond.reset_index(drop=True), ct.reset_index(drop=True)


def run_case(
    case: CaseSpec,
    *,
    epochs: int = 100,
    subsample_cells: int = 3000,
    n_top_genes: int = 2000,
    top_k_genes: int = 5,
    silent: bool = True,
) -> dict:
    """Train the case-spec encoder, compute UMAP and the bio-validation payload."""
    if not case.h5ad_path.exists():
        raise FileNotFoundError(f"missing h5ad for case {case.case_id}: {case.h5ad_path}")

    log.info("[%s] loading %s", case.case_id, case.h5ad_path)
    adata = ad.read_h5ad(str(case.h5ad_path))
    n_obs_raw = adata.n_obs

    # Preprocess (QC + subsample + log + HVG) — same path as the production
    # encoder sweep so the latent here is comparable to the Axis A row.
    adata_pp = preprocess_scrna_scccvgben(
        adata,
        subsample_cells=subsample_cells,
        n_top_genes=n_top_genes,
    )
    log.info("[%s] preprocessed: %s", case.case_id, adata_pp.shape)

    # Train encoder via the standard runner
    import torch
    from scccvgben.external.reference_core.cgvae import CGVAE_agent
    from scccvgben.training.scccvgben_runner import SCCCVGBEN_DEFAULTS

    cfg = {**SCCCVGBEN_DEFAULTS}
    cfg["graph_type"] = case.encoder
    cfg.pop("subsample_cells", None)

    cfg.pop("n_top_genes", None)
    cfg.pop("epochs", None)
    cfg.pop("device", None)
    cfg.pop("tech", None)
    cfg.pop("encoder_type", None)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    agent = CGVAE_agent(adata=adata_pp, layer="counts", **cfg, device=device)
    agent.fit(epochs=epochs, silent=silent)
    latent = np.asarray(agent.get_latent())
    log.info("[%s] trained latent: %s", case.case_id, latent.shape)

    # UMAP from latent (drives most panels)
    adata_pp.obsm["X_latent"] = latent
    sc.pp.neighbors(adata_pp, use_rep="X_latent", n_neighbors=15)
    sc.tl.umap(adata_pp, random_state=42)
    umap = np.asarray(adata_pp.obsm["X_umap"])

    # Labels
    condition, cell_type = _ensure_labels(adata_pp, case)
    pseudotime = None
    if case.pseudotime_obs and case.pseudotime_obs in adata_pp.obs.columns:
        pseudotime = adata_pp.obs[case.pseudotime_obs].to_numpy(dtype=float)

    # Latent self-correlation
    latent_corr = latent_self_correlation(latent)

    # Top-K genes per dim (use log-normalised X, which is adata_pp.X)
    X = adata_pp.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    expression_df = pd.DataFrame(np.asarray(X), columns=list(adata_pp.var_names))
    top_k = top_k_genes_per_dim(latent, expression_df, k=top_k_genes, method="spearman")
    # Subset expression matrix down to genes mentioned in top-k for compose
    keep_genes = list(dict.fromkeys(top_k["gene"]))
    expression_subset = expression_df[keep_genes]

    return {
        "case": case,
        "n_obs": int(n_obs_raw),
        "n_obs_subsampled": int(adata_pp.n_obs),
        "n_vars_post_hvg": int(adata_pp.n_vars),
        "umap": umap,
        "latent": latent,
        "condition": condition,
        "cell_type": cell_type,
        "pseudotime": pseudotime,
        "latent_corr": latent_corr,
        "top_k_genes_df": top_k,
        "expression": expression_subset,
    }
