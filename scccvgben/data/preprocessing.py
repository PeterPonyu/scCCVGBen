"""Preprocessing for scRNA-seq and scATAC-seq AnnData objects."""

from __future__ import annotations

import logging

import numpy as np
import anndata as ad
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse

log = logging.getLogger(__name__)


def preprocess_scrna(
    adata: ad.AnnData,
    n_top_genes: int = 5000,
    n_pcs: int = 50,
    subsample: int | None = None,
) -> ad.AnnData:
    """Preprocess scRNA-seq AnnData.

    Pipeline: (optional subsample) -> drop zero-count cells/genes ->
              library-size norm -> log1p -> HVG selection -> PCA.

    The leading cell/gene filter prevents degenerate gene means (e.g. identical
    zero-means or NaNs from ``log1p`` on synthetic data) from crashing scanpy's
    ``highly_variable_genes`` binning. If cell_ranger-flavor HVG still fails on
    pathological inputs, we fall back to ``flavor="seurat"``.
    """
    adata = adata.copy()

    if subsample is not None and adata.n_obs > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, size=subsample, replace=False)
        adata = adata[idx].copy()

    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_cells=1)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_top_genes = min(n_top_genes, adata.n_vars)
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes, flavor="cell_ranger", subset=True
        )
    except (ValueError, Exception) as exc:
        log.warning(
            "cell_ranger HVG failed (%s); falling back to seurat flavor.", exc
        )
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes, flavor="seurat", subset=True
        )

    sc.pp.pca(adata, n_comps=min(n_pcs, max(1, adata.n_vars - 1)))

    return adata


def preprocess_scatac(
    adata: ad.AnnData,
    n_top_peaks: int = 2000,
    n_lsi: int = 50,
    subsample: int | None = None,
) -> ad.AnnData:
    """Preprocess scATAC-seq AnnData.

    Pipeline: (optional subsample) -> TF-IDF -> top HV peaks (by variance) ->
              TruncatedSVD / LSI.

    Returns AnnData with .obsm['X_lsi'] set and .X = TF-IDF matrix.
    """
    adata = adata.copy()

    if subsample is not None and adata.n_obs > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, size=subsample, replace=False)
        adata = adata[idx].copy()

    # TF-IDF
    X = adata.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float64)
    tf = X / (X.sum(axis=1, keepdims=True) + 1e-10)
    idf = np.log1p(X.shape[0] / (X.sum(axis=0) + 1))
    tfidf = tf * idf

    # Select top HV peaks by variance
    var = tfidf.var(axis=0)
    top_idx = np.argsort(var)[::-1][:n_top_peaks]
    tfidf_hv = tfidf[:, top_idx]
    adata = adata[:, top_idx].copy()
    adata.X = tfidf_hv.astype(np.float32)

    # LSI via TruncatedSVD (skip first component — correlated with library size)
    n_components = min(n_lsi + 1, tfidf_hv.shape[1] - 1, tfidf_hv.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsi = svd.fit_transform(tfidf_hv)[:, 1:]   # drop first component

    adata.obsm["X_lsi"] = lsi.astype(np.float32)
    return adata
