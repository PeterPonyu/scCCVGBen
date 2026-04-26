"""Compute the benchmark metrics for scCCVGBen runs.

The active schema keeps annotation-free clustering compactness scores, DRE
neighbourhood-preservation diagnostics, and LSE intrinsic-geometry outputs.
Two legacy label-agreement fields were removed on 2026-04-25 because most
source datasets lack metric-grade biological labels; the historical fallback
therefore produced self-comparisons rather than informative benchmark scores.

``COR`` is also omitted from active outputs because current sweeps did not
populate it consistently and its direction/definition differed from the
vendored reference. The methodologically aligned latent-dimension
decorrelation helper is preserved as an opt-in utility, but active runners do
not call it and do not recompute historical results.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.cluster import KMeans

# Vendored reference core evaluators (bit-for-bit copy)
from scccvgben.external.reference_core.dre import evaluate_dimensionality_reduction
from scccvgben.external.reference_core.lse import evaluate_single_cell_latent_space

# Canonical 24-column schema (method + 23 metrics). legacy label-agreement fields and COR dropped —
# see module docstring for rationale.
METRIC_COLS = [
    "method",
    "ASW", "DAV", "CAL",
    "distance_correlation_umap", "Q_local_umap", "Q_global_umap",
    "K_max_umap", "overall_quality_umap",
    "distance_correlation_tsne", "Q_local_tsne", "Q_global_tsne",
    "K_max_tsne", "overall_quality_tsne",
    "manifold_dimensionality_intrin", "spectral_decay_rate_intrin",
    "participation_ratio_intrin", "anisotropy_score_intrin",
    "trajectory_directionality_intrin", "noise_resilience_intrin",
    "core_quality_intrin", "overall_quality_intrin",
    "data_type_intrin", "interpretation_intrin",
]
# Back-compat alias (some downstream scripts still import METRIC_COLUMNS)
METRIC_COLUMNS = METRIC_COLS


def _reference_compute_metrics(latent: np.ndarray,
                            labels: np.ndarray | None,
                            data_type: str = "trajectory") -> dict:
    """Verbatim port of the reference benchmark compute_metrics routine.

    Returns a flat dict keyed by the exact column names used in CG_dl_merged
    (method column is injected by the wrapper).
    """
    results: dict = {}

    # Clustering compactness metrics (no ground-truth dependency) -------------
    # ``labels`` is accepted only to size n_clusters; values themselves are
    # never compared (legacy label-agreement fields removed — see module docstring).
    if labels is not None:
        n_clusters = max(2, int(len(np.unique(labels))))
    else:
        n_clusters = 10

    if latent.shape[0] < 2 or np.unique(latent, axis=0).shape[0] < 2:
        for k in ("ASW", "DAV", "CAL"):
            results[k] = np.nan
    else:
        km = KMeans(
            n_clusters=min(n_clusters, latent.shape[0] - 1),
            n_init=10, random_state=42,
        ).fit(latent)
        pred = km.labels_
        try:
            results["ASW"] = float(silhouette_score(latent, pred))
        except Exception:
            results["ASW"] = np.nan
        try:
            results["DAV"] = float(davies_bouldin_score(latent, pred))
        except Exception:
            results["DAV"] = np.nan
        try:
            results["CAL"] = float(calinski_harabasz_score(latent, pred))
        except Exception:
            results["CAL"] = np.nan

    # DRE UMAP + tSNE via scanpy (same path as the reference routine) --------
    try:
        import anndata as ad
        adata_tmp = ad.AnnData(X=latent.astype(np.float32))
        sc.pp.neighbors(adata_tmp, use_rep="X", n_neighbors=min(15, max(2, latent.shape[0] - 1)))
        sc.tl.umap(adata_tmp, random_state=42)
        sc.tl.tsne(adata_tmp, use_rep="X", random_state=42)
        X_umap = adata_tmp.obsm["X_umap"]
        X_tsne = adata_tmp.obsm["X_tsne"]

        res_umap = evaluate_dimensionality_reduction(latent, X_umap, verbose=False)
        for k, v in res_umap.items():
            results[f"{k}_umap"] = v
        res_tsne = evaluate_dimensionality_reduction(latent, X_tsne, verbose=False)
        for k, v in res_tsne.items():
            results[f"{k}_tsne"] = v
        del adata_tmp
    except Exception as e:
        warnings.warn(f"DRE failed: {e}")
        for k in ("distance_correlation", "Q_local", "Q_global", "K_max", "overall_quality"):
            results[f"{k}_umap"] = np.nan
            results[f"{k}_tsne"] = np.nan

    # LSE intrinsic (9 metrics from full SingleCellLatentSpaceEvaluator) -------
    try:
        res_intrin = evaluate_single_cell_latent_space(latent, data_type=data_type, verbose=False)
        for k, v in res_intrin.items():
            results[f"{k}_intrin"] = v
    except Exception as e:
        warnings.warn(f"LSE failed: {e}")
        for k in ("manifold_dimensionality", "spectral_decay_rate", "participation_ratio",
                  "anisotropy_score", "trajectory_directionality", "noise_resilience",
                  "core_quality", "overall_quality", "data_type", "interpretation"):
            results[f"{k}_intrin"] = np.nan

    return results


def compute_metrics(Z: np.ndarray,
                    X_orig: np.ndarray | None = None,
                    labels: np.ndarray | None = None,
                    method_name: str = "scCCVGBen",
                    data_type: str = "trajectory") -> pd.DataFrame:
    """Wrapper: return a 1-row DataFrame with METRIC_COLS column order.

    Parameters
    ----------
    Z          : (N, L) latent embedding
    X_orig     : retained for legacy keyword compatibility; ignored. The COR
                 column it used to populate has been removed (see module
                 docstring; ``compute_latent_dimension_decorrelation`` is
                 the methodologically correct replacement).
    labels     : retained for legacy keyword compatibility; not consulted by
                 the active metric set (see module docstring).
    method_name: string for the 'method' column
    data_type  : 'trajectory' (default) or 'steady_state' for LSE scoring
    """
    del X_orig  # explicitly ignored — see docstring
    Z = np.asarray(Z, dtype=np.float32)
    results = _reference_compute_metrics(Z, labels, data_type=data_type)
    results["method"] = method_name

    # Ensure every canonical column present
    for col in METRIC_COLS:
        results.setdefault(col, np.nan)
    return pd.DataFrame([{col: results.get(col, np.nan) for col in METRIC_COLS}])
