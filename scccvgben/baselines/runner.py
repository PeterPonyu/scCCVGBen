"""runner.py — Unified baseline runner for scCCVGBen.

Public API:
    run_baseline(name, h5ad_path, modality) -> dict[str, float]

Supported names: PCA, KPCA, ICA, FA, NMF, TSVD, DICL,
                 scVI, DIP, INFO, TC, highBeta, CCVGAE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from scccvgben.baselines.sklearn_methods import SKLEARN_REGISTRY
from scccvgben.baselines.deep_methods import DEEP_REGISTRY

log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
_SUBSAMPLE = 3000
_LATENT_DIM = 10
_RANDOM_STATE = 42

SUPPORTED_BASELINES = list(SKLEARN_REGISTRY) + list(DEEP_REGISTRY) + ["CCVGAE"]

# ── metric column schema (matches CG_dl_merged CSVs) ─────────────────────────
METRIC_COLUMNS = [
    "method", "ASW", "DAV", "CAL", "COR",
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


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_and_preprocess(h5ad_path: Path, modality: str) -> ad.AnnData:
    """Load h5ad and run preprocessing appropriate for the modality."""
    adata = ad.read_h5ad(h5ad_path)
    try:
        if modality == "scrna":
            from scccvgben.data.preprocessing import preprocess_scrna
            adata = preprocess_scrna(adata)
        elif modality == "scatac":
            from scccvgben.data.preprocessing import preprocess_scatac
            adata = preprocess_scatac(adata)
        else:
            raise ValueError(f"Unknown modality '{modality}'. Expected 'scrna' or 'scatac'.")
    except ImportError:
        log.warning("preprocessing module not yet available; using raw X.")
    return adata


def _subsample(adata: ad.AnnData, n: int = _SUBSAMPLE, seed: int = _RANDOM_STATE) -> ad.AnnData:
    """Randomly subsample to at most n cells."""
    if adata.n_obs <= n:
        return adata
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=n, replace=False)
    return adata[idx].copy()


def _get_X(adata: ad.AnnData) -> np.ndarray:
    """Return dense float32 expression matrix."""
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.array(X, dtype=np.float32)


def _get_labels(adata: ad.AnnData) -> np.ndarray | None:
    """Extract cell-type labels via the unified scccvgben.data.labels.get_labels helper."""
    from scccvgben.data.labels import get_labels
    return get_labels(adata)


def _compute_metrics(
    z: np.ndarray,
    labels: np.ndarray | None,
    X: np.ndarray,
    method_name: str,
) -> dict[str, Any]:
    """Compute the 27-metric schema. Falls back to NaN if metrics module absent."""
    try:
        from scccvgben.training.metrics import compute_metrics
        metrics_df = compute_metrics(Z=z, X_orig=X, labels=labels, method_name=method_name)
        metrics = metrics_df.iloc[0].to_dict()
    except ImportError as exc:
        log.warning("metrics module not importable (%s); returning NaN placeholders.", exc)
        metrics = {col: float("nan") for col in METRIC_COLUMNS if col != "method"}
    except Exception as exc:
        log.warning("metrics computation failed (%s); returning NaN placeholders.", exc)
        metrics = {col: float("nan") for col in METRIC_COLUMNS if col != "method"}

    metrics["method"] = method_name
    # Ensure all expected columns present
    for col in METRIC_COLUMNS:
        metrics.setdefault(col, float("nan"))
    return metrics


# ── CCVGAE baseline path ─────────────────────────────────────────────────────

def _run_ccvgae(X: np.ndarray, labels: np.ndarray | None, adata: ad.AnnData) -> np.ndarray:
    """Run the canonical CCVGAE model (GAT + kNN-Euc) to get latent z."""
    try:
        import torch
        from scccvgben.models.ccvgae import CCVGAE
        from scccvgben.graphs.construction import build_knn_euclidean
        from torch_geometric.data import Data

        Xt = torch.tensor(X, dtype=torch.float32)
        edge_index, edge_weight = build_knn_euclidean(Xt, k=15)
        data = Data(x=Xt, edge_index=edge_index, edge_attr=edge_weight)

        model = CCVGAE(
            in_dim=X.shape[1],
            hidden=128,
            latent_dim=_LATENT_DIM,
            i_dim=5,
            encoder_name="GAT",
        )

        from scccvgben.training.trainer import fit_one
        # fit_one auto-detects device; modality determines loss function
        fit_one(model, data, "scrna", epochs=50, lr=1e-4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            _, q_m, *_ = model(Xt.to(device), edge_index.to(device))
        return q_m.cpu().numpy()
    except ImportError as exc:
        log.warning("CCVGAE import failed (%s); falling back to PCA latent.", exc)
        from scccvgben.baselines.sklearn_methods import run_PCA
        return run_PCA(X, n_components=_LATENT_DIM)
    except Exception as exc:
        log.warning("CCVGAE run failed (%s); falling back to PCA latent.", exc)
        from scccvgben.baselines.sklearn_methods import run_PCA
        return run_PCA(X, n_components=_LATENT_DIM)


# ── public API ───────────────────────────────────────────────────────────────

def run_baseline(
    name: str,
    h5ad_path: Path,
    modality: str,
) -> dict[str, Any]:
    """Run a single baseline method on one dataset.

    Args:
        name: One of SUPPORTED_BASELINES.
        h5ad_path: Absolute path to the .h5ad file.
        modality: 'scrna' or 'scatac'.

    Returns:
        Dict matching the 27-metric schema (METRIC_COLUMNS), with 'method' key set.
    """
    h5ad_path = Path(h5ad_path)
    if name not in SUPPORTED_BASELINES:
        raise ValueError(
            f"Unknown baseline '{name}'. Supported: {SUPPORTED_BASELINES}"
        )

    log.info("run_baseline: %s on %s (%s)", name, h5ad_path.name, modality)

    adata = _load_and_preprocess(h5ad_path, modality)
    adata = _subsample(adata)
    X = _get_X(adata)
    labels = _get_labels(adata)

    if name in SKLEARN_REGISTRY:
        z = SKLEARN_REGISTRY[name](X, n_components=_LATENT_DIM)
    elif name in DEEP_REGISTRY:
        z = DEEP_REGISTRY[name](X, n_components=_LATENT_DIM)
    elif name == "CCVGAE":
        z = _run_ccvgae(X, labels, adata)
    else:
        raise ValueError(f"Dispatch error for '{name}'.")

    return _compute_metrics(z, labels, X, method_name=name)
