"""Unified label extraction for scCCVGBen — descriptive / plotting use only.

DO NOT USE as metric-grade ground truth (deprecated 2026-04-25)
===========================================================
The current self-supervised benchmark protocol defines reference labels as
``KMeans(n_clusters=latent_dim).fit_predict(X_pre)`` on the pre-processed
input matrix, **not** any column of ``adata.obs``. See
``scccvgben.training.metrics`` module docstring for the current
self-supervised metric protocol.

This module is retained for legitimate descriptive uses — e.g. colouring a
UMAP plot by ``batch`` or ``celltype`` when those columns happen to exist —
but its output must not be passed as a ground-truth label to clustering
metrics. The fallback chain below is intentionally permissive (returning
``batch``, ``Sex``, etc. when present) and is therefore not safe as a
metric-grade label source.

Usage:
    labels = get_labels(adata)  # returns str-array or None
"""

from __future__ import annotations

import anndata as ad
import numpy as np


_PRIMARY_COLS = (
    "cell_type",
    "celltype",
    "label",
    "labels",
    "cluster",
    "clusters",
    "annotation",
    "CellType",
)


def get_labels(adata: ad.AnnData) -> np.ndarray | None:
    """Return cell labels as a string array, or ``None`` if no suitable column exists.

    Resolution order (mirrors the reference benchmark methodology):
    1. Primary annotation columns: ``cell_type``, ``celltype``, ``label``,
       ``labels``, ``cluster``, ``clusters``, ``annotation``, ``CellType``.
    2. Last-resort: the first ``obs`` column with pandas category dtype
       or fewer than 100 unique values.
    """
    for col in _PRIMARY_COLS:
        if col in adata.obs.columns:
            return adata.obs[col].astype(str).values

    for col in adata.obs.columns:
        dtype_name = adata.obs[col].dtype.name
        if dtype_name == "category" or adata.obs[col].nunique(dropna=True) < 100:
            return adata.obs[col].astype(str).values

    return None
