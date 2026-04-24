"""Unified label extraction for scCCVGBen.

Ported from /home/zeyufu/LAB/CCVGAE/CCVGAE_supplement/run_hyperparam_sensitivity.py
(revised 2026-04-23). Centralises the fallback chain that ``loader.py`` and
``baselines/runner.py`` previously implemented independently, preventing
drift between the two code paths.

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

    Resolution order (mirrors CCVGAE revised methodology):
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
