"""Compute the 27-column benchmark metrics matching CG_dl_merged/ schema.

This module is a thin wrapper over the vendored reference benchmark
``compute_metrics`` logic. Keep benchmark rows bit-compatible with the reused
CG_dl_merged CSVs.
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
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import KMeans

# Vendored reference core evaluators (bit-for-bit copy)
from scccvgben.external.reference_core.dre import evaluate_dimensionality_reduction
from scccvgben.external.reference_core.lse import evaluate_single_cell_latent_space

# Canonical 27-column schema (method + 26 metrics). Matches CG_dl_merged
# exactly so a new row can be pd.concat'ed onto a reused CSV.
METRIC_COLS = [
    "method",
    "ASW", "DAV", "CAL", "COR",
    "distance_correlation_umap", "Q_local_umap", "Q_global_umap",
    "K_max_umap", "overall_quality_umap",
    "distance_correlation_tsne", "Q_local_tsne", "Q_global_tsne",
    "K_max_tsne", "overall_quality_tsne",
    "manifold_dimensionality_intrin", "spectral_decay_rate_intrin",
    "participation_ratio_intrin", "anisotropy_score_intrin",
    "trajectory_directionality_intrin", "noise_resilience_intrin",
    "core_quality_intrin", "overall_quality_intrin",
    "data_type_intrin", "interpretation_intrin",
    "NMI", "ARI",
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

    # BEN clustering metrics ---------------------------------------------------
    if labels is not None:
        n_clusters = max(2, int(len(np.unique(labels))))
    else:
        n_clusters = 10

    if latent.shape[0] < 2 or np.unique(latent, axis=0).shape[0] < 2:
        for k in ("NMI", "ARI", "ASW", "DAV", "CAL"):
            results[k] = np.nan
    else:
        km = KMeans(
            n_clusters=min(n_clusters, latent.shape[0] - 1),
            n_init=10, random_state=42,
        ).fit(latent)
        pred = km.labels_
        ref = labels if labels is not None else pred
        results["NMI"] = float(normalized_mutual_info_score(ref, pred))
        results["ARI"] = float(adjusted_rand_score(ref, pred))
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

    # COR (Spearman on pairwise distances Z vs X_orig) — scCCVGBen doesn't include
    # this in hyperparam_sensitivity but CG_dl_merged has a 'COR' column. Leave
    # as NaN here; callers that need it should populate externally before write.
    results.setdefault("COR", np.nan)
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
    X_orig     : (N, F) pre-processed feature matrix. If supplied, Spearman
                 correlation between pairwise distances in Z vs X is stored
                 under 'COR'. If None, COR is NaN.
    labels     : (N,) integer cell-type labels or None (self-reference mode)
    method_name: string for the 'method' column
    data_type  : 'trajectory' (default) or 'steady_state' for LSE scoring
    """
    Z = np.asarray(Z, dtype=np.float32)
    results = _reference_compute_metrics(Z, labels, data_type=data_type)
    results["method"] = method_name

    # Optional COR via spearman on pairwise distances
    if X_orig is not None and np.isnan(results.get("COR", np.nan)):
        try:
            from sklearn.metrics.pairwise import pairwise_distances
            from scipy.stats import spearmanr
            n = min(500, Z.shape[0])
            rng = np.random.default_rng(0)
            idx = rng.choice(Z.shape[0], size=n, replace=False)
            dz = pairwise_distances(Z[idx]).ravel()
            dx = pairwise_distances(np.asarray(X_orig)[idx]).ravel()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, _ = spearmanr(dz, dx)
            results["COR"] = float(r) if np.isfinite(r) else np.nan
        except Exception:
            results["COR"] = np.nan

    # Ensure every canonical column present
    for col in METRIC_COLS:
        results.setdefault(col, np.nan)
    return pd.DataFrame([{col: results.get(col, np.nan) for col in METRIC_COLS}])
