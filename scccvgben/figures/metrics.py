"""Shared metric catalog and display helpers for manuscript figures.

The benchmark computes 26 reported fields. Twenty-four are numeric and can be
drawn as box/strip, heatmap, or delta panels; the remaining two intrinsic-space
fields are categorical/structured annotations and should be reported in text or
tables rather than coerced into numeric plots.
"""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike

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

METRIC_TO_FAMILY: dict[str, str] = {
    metric: family
    for family, metrics in METRIC_FAMILY_ROWS
    for metric in metrics
}

METRIC_FAMILY_TITLES: dict[str, str] = {
    "BEN": "clustering\nlabels + distance",
    "DRE-UMAP": "UMAP\nneighbourhoods",
    "DRE-tSNE": "t-SNE\nneighbourhoods",
    "LSE": "intrinsic\nlatent geometry",
}

NUMERIC_METRICS: tuple[str, ...] = tuple(
    metric for _, metrics in NUMERIC_METRIC_FAMILIES for metric in metrics
)

METRIC_PANEL_GRID: tuple[tuple[str, ...], ...] = tuple(
    tuple(NUMERIC_METRICS[i:i + 6]) for i in range(0, len(NUMERIC_METRICS), 6)
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


def metric_coverage_audit(
    df: pd.DataFrame,
    metrics: Iterable[str] = NUMERIC_METRICS,
    *,
    figure_id: str,
    group_col: str = "method",
    metric_col: str = "metric",
    dataset_col: str = "dataset_id",
    expected_datasets: int | None = None,
    expected_methods: int | None = None,
) -> pd.DataFrame:
    """Return one audit row per expected metric.

    The long-form plotting tables intentionally contain only numeric values, so
    a metric with zero rows means either the source column is absent or every
    value was non-numeric/empty.  Both cases must remain visible in publication
    figures as an explicit missing panel rather than being silently dropped.
    """
    metric_list = list(metrics)
    expected_methods = (
        expected_methods
        if expected_methods is not None
        else (
            int(df[group_col].dropna().nunique())
            if group_col in df.columns and not df.empty
            else 0
        )
    )
    rows: list[dict[str, object]] = []
    for order, metric in enumerate(metric_list, start=1):
        sub = (
            df[df[metric_col].astype(str) == metric]
            if metric_col in df.columns and not df.empty
            else pd.DataFrame()
        )
        n_values = int(len(sub))
        n_datasets = (
            int(sub[dataset_col].dropna().nunique())
            if dataset_col in sub.columns and not sub.empty
            else 0
        )
        n_methods = (
            int(sub[group_col].dropna().nunique())
            if group_col in sub.columns and not sub.empty
            else 0
        )
        if n_values == 0:
            status = "missing"
        elif (
            (expected_datasets is not None and n_datasets < expected_datasets)
            or (expected_methods > 0 and n_methods < expected_methods)
        ):
            status = "partial"
        else:
            status = "complete"
        rows.append(
            {
                "figure_id": figure_id,
                "metric_order": order,
                "metric": metric,
                "family": METRIC_TO_FAMILY.get(metric, "unknown"),
                "status": status,
                "value_rows": n_values,
                "dataset_count": n_datasets,
                "expected_dataset_count": expected_datasets,
                "method_count": n_methods,
                "expected_method_count": expected_methods,
            }
        )
    return pd.DataFrame(rows)


def write_metric_audit(
    path: str | PathLike[str],
    audit: pd.DataFrame,
    *,
    figure_id: str,
) -> None:
    """Upsert a figure's metric audit rows into a CSV file."""
    from pathlib import Path

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if "figure_id" in existing.columns:
            existing = existing[existing["figure_id"].astype(str) != figure_id]
        audit = pd.concat([existing, audit], ignore_index=True)
    audit.to_csv(out_path, index=False)


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
