"""Shared metric catalog and display helpers for manuscript figures.

The benchmark computes 26 reported fields. Twenty-four are numeric and can be
drawn as box/strip, heatmap, or delta panels; the remaining two intrinsic-space
fields are categorical/structured annotations and should be reported in text or
tables rather than coerced into numeric plots.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

CLUSTERING_METRICS: tuple[str, ...] = (
    "ASW",
    "DAV",
    "CAL",
    "NMI",
    "ARI",
    "COR",
)

DRE_UMAP_METRICS: tuple[str, ...] = (
    "distance_correlation_umap",
    "Q_local_umap",
    "Q_global_umap",
    "K_max_umap",
    "overall_quality_umap",
)

DRE_TSNE_METRICS: tuple[str, ...] = (
    "distance_correlation_tsne",
    "Q_local_tsne",
    "Q_global_tsne",
    "K_max_tsne",
    "overall_quality_tsne",
)

INTRINSIC_NUMERIC_METRICS: tuple[str, ...] = (
    "manifold_dimensionality_intrin",
    "spectral_decay_rate_intrin",
    "participation_ratio_intrin",
    "anisotropy_score_intrin",
    "trajectory_directionality_intrin",
    "noise_resilience_intrin",
    "core_quality_intrin",
    "overall_quality_intrin",
)

NON_NUMERIC_METRICS: tuple[str, ...] = (
    "data_type_intrin",
    "interpretation_intrin",
)

NUMERIC_METRIC_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Clustering and distance preservation", CLUSTERING_METRICS),
    ("UMAP neighbourhood preservation", DRE_UMAP_METRICS),
    ("t-SNE neighbourhood preservation", DRE_TSNE_METRICS),
    ("Intrinsic latent geometry", INTRINSIC_NUMERIC_METRICS),
)

METRIC_FAMILY_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("BEN", CLUSTERING_METRICS),
    ("DRE-UMAP", DRE_UMAP_METRICS),
    ("DRE-tSNE", DRE_TSNE_METRICS),
    ("LSE", INTRINSIC_NUMERIC_METRICS),
)

METRIC_FAMILY_TITLES: dict[str, str] = {
    "BEN": "clustering\nlabels + distance",
    "DRE-UMAP": "UMAP\nneighbourhoods",
    "DRE-tSNE": "t-SNE\nneighbourhoods",
    "LSE": "intrinsic\nlatent geometry",
}

NUMERIC_METRICS: tuple[str, ...] = tuple(
    metric for _, metrics in NUMERIC_METRIC_FAMILIES for metric in metrics
)

LOWER_IS_BETTER: frozenset[str] = frozenset({"DAV"})

METRIC_LABELS: dict[str, str] = {
    "ASW": "ASW ↑",
    "DAV": "DAV ↓",
    "CAL": "CAL ↑",
    "NMI": "NMI ↑",
    "ARI": "ARI ↑",
    "COR": "COR ↑",
    "distance_correlation_umap": "DC UMAP ↑",
    "Q_local_umap": "QL UMAP ↑",
    "Q_global_umap": "QG UMAP ↑",
    "K_max_umap": "Kmax UMAP",
    "overall_quality_umap": "Overall UMAP ↑",
    "distance_correlation_tsne": "DC t-SNE ↑",
    "Q_local_tsne": "QL t-SNE ↑",
    "Q_global_tsne": "QG t-SNE ↑",
    "K_max_tsne": "Kmax t-SNE",
    "overall_quality_tsne": "Overall t-SNE ↑",
    "manifold_dimensionality_intrin": "Manifold dim. ↑",
    "spectral_decay_rate_intrin": "Spectral decay ↑",
    "participation_ratio_intrin": "Part. ratio ↑",
    "anisotropy_score_intrin": "Anisotropy ↑",
    "trajectory_directionality_intrin": "Trajectory dir. ↑",
    "noise_resilience_intrin": "Noise resil. ↑",
    "core_quality_intrin": "Core quality ↑",
    "overall_quality_intrin": "Intrinsic overall ↑",
}


def available_numeric_metrics(
    df: pd.DataFrame,
    metrics: Iterable[str] = NUMERIC_METRICS,
    *,
    metric_col: str = "metric",
) -> list[str]:
    """Return catalog metrics present in a long-form metric DataFrame."""
    present = set(df[metric_col].dropna().astype(str)) if metric_col in df.columns else set()
    return [metric for metric in metrics if metric in present]


def short_method_name(method: str) -> str:
    """Short, publication-friendly labels while keeping scCCVGBen explicit."""
    replacements = {
        "scCCVGBen_GAT_kNN_euc": "kNN-euc",
        "scCCVGBen_GAT_kNN_cosine": "kNN-cos",
        "scCCVGBen_GAT_snn": "SNN",
        "scCCVGBen_GAT_mutual_knn": "Mutual-kNN",
        "scCCVGBen_GAT_gaussian_threshold": "Gaussian",
        "scCCVGBen_GAT": "GAT",
        "scCCVGBen_GATv2": "GATv2",
        "scCCVGBen_Transformer": "Transformer",
        "scCCVGBen_SuperGAT": "SuperGAT",
        "scCCVGBen_GCN": "GCN",
        "scCCVGBen_SAGE": "SAGE",
        "scCCVGBen_GIN": "GIN",
        "scCCVGBen_Cheb": "Cheb",
        "scCCVGBen_EdgeConv": "EdgeConv",
        "scCCVGBen_ARMA": "ARMA",
        "scCCVGBen_SG": "SG",
        "scCCVGBen_TAG": "TAG",
        "scCCVGBen_Graph": "Graph",
        "scCCVGBen_SSG": "SSG",
        "scCCVGBen": "scCCVGBen",
        "DIPVAE": "DIP",
        "InfoVAE": "INFO",
        "TCVAE": "TC",
        "HighBetaVAE": "highBeta",
    }
    return replacements.get(method, method)


def add_method_display(long_df: pd.DataFrame, *, source_col: str = "method") -> pd.DataFrame:
    """Return a copy with ``method_display`` labels for dense x axes."""
    out = long_df.copy()
    out["method_display"] = out[source_col].astype(str).map(short_method_name)
    return out
