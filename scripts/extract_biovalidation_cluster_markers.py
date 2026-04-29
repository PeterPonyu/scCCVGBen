#!/usr/bin/env python3
"""Extract no-training marker genes for inferred bio-validation clusters.

This script intentionally stops at preprocessing, Leiden clustering, and
ranked marker extraction.  It never calls ``run_case`` and never instantiates
or fits the CCVGAE model.  The resulting marker tables are used to build the
auditable biological label map for the manuscript case-study figures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scccvgben.biovalidation.case_definition import CASES, CaseSpec  # noqa: E402


DEFAULT_CASES = ("GASTRIC", "UCB", "HSC_AGE", "COVID")
DEFAULT_OUTDIR = REPO_ROOT / ".omx" / "reports" / "case-study-celltype-labels"
LEADER_REPO_ROOT = Path(os.environ.get("SCCCVGBEN_LEADER_REPO_ROOT", str(REPO_ROOT))).expanduser()
LEGACY_RESULTS = Path(
    os.environ.get("SCCCVGBEN_LEGACY_RESULTS", str(REPO_ROOT.parent / "CCVGAE" / "CG_results"))
).expanduser()


def _parse_cases(value: str) -> list[str]:
    case_ids = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = [case_id for case_id in case_ids if case_id not in CASES]
    if unknown:
        raise SystemExit(f"unknown case id(s): {', '.join(unknown)}")
    return case_ids


def _resolve_h5ad(case: CaseSpec) -> Path:
    """Resolve data paths inside a team worktree without mutating inputs.

    Team worktrees contain source files but not necessarily the large local
    h5ad symlinks.  When the case path is absent in the worker checkout, fall
    back to the leader checkout and the legacy CCVGAE results directory.
    """
    candidates = [case.h5ad_path]
    try:
        rel = case.h5ad_path.relative_to(REPO_ROOT)
        candidates.append(LEADER_REPO_ROOT / rel)
    except ValueError:
        pass
    if case.h5ad_path.name.startswith("SubSampled"):
        candidates.extend(
            [
                LEGACY_RESULTS / case.h5ad_path.name,
                LEADER_REPO_ROOT / "CCVGAE" / "CG_results" / case.h5ad_path.name,
            ]
        )
    else:
        candidates.append(LEADER_REPO_ROOT / "workspace" / "data" / "scrna" / case.h5ad_path.name)

    for path in candidates:
        if path.exists():
            return path
    tried = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing h5ad for {case.case_id}; tried:\n  - {tried}")


def _load_subsampled_h5ad(path: Path, *, subsample_cells: int, random_state: int) -> ad.AnnData:
    """Read an h5ad, sampling rows first for very large backed inputs."""
    backed = ad.read_h5ad(path, backed="r")
    n_obs = backed.n_obs
    if n_obs > subsample_cells:
        rng = np.random.default_rng(random_state)
        obs_idx = np.sort(rng.choice(n_obs, size=subsample_cells, replace=False))
        adata = backed[obs_idx, :].to_memory()
    else:
        adata = backed.to_memory()
    backed.file.close()
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def _preprocess_for_markers(
    adata: ad.AnnData,
    *,
    n_top_genes: int,
    min_counts_per_cell: int,
    min_cells_per_gene: int,
) -> tuple[ad.AnnData, str]:
    """Apply the same no-model preprocessing contract used before clustering."""
    if min_counts_per_cell > 0:
        sc.pp.filter_cells(adata, min_counts=min_counts_per_cell)
    if min_cells_per_gene > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    hvg_flavor = "seurat_v3"
    n_top = min(n_top_genes, max(2, adata.n_vars - 1))
    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top,
            flavor=hvg_flavor,
            layer="counts",
        )
    except Exception as exc:  # pragma: no cover - environment fallback
        hvg_flavor = f"cell_ranger_fallback_after_{type(exc).__name__}"
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="cell_ranger")
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata, hvg_flavor


def _ensure_cell_type_clusters(adata: ad.AnnData) -> pd.Series:
    """Match the cell-type fallback: PCA -> neighbors -> Leiden resolution 0.8."""
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=min(20, adata.n_vars - 1))
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)
    sc.tl.leiden(
        adata,
        resolution=0.8,
        key_added="_ct_leiden",
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    return adata.obs["_ct_leiden"].astype(str)


def _extract_case_markers(
    case_id: str,
    *,
    subsample_cells: int,
    n_top_genes: int,
    top_markers: int,
    random_state: int,
    min_counts_per_cell: int,
    min_cells_per_gene: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    case = CASES[case_id]
    path = _resolve_h5ad(case)
    raw = ad.read_h5ad(path, backed="r")
    raw_shape = tuple(map(int, raw.shape))
    raw.file.close()

    adata = _load_subsampled_h5ad(path, subsample_cells=subsample_cells, random_state=random_state)
    sampled_n_obs = int(adata.n_obs)
    adata, hvg_flavor = _preprocess_for_markers(
        adata,
        n_top_genes=n_top_genes,
        min_counts_per_cell=min_counts_per_cell,
        min_cells_per_gene=min_cells_per_gene,
    )
    cluster_series = _ensure_cell_type_clusters(adata)
    adata.obs["_ct_leiden"] = cluster_series.astype("category")
    cluster_counts = cluster_series.value_counts().sort_index()

    sc.tl.rank_genes_groups(
        adata,
        groupby="_ct_leiden",
        method="wilcoxon",
        n_genes=min(top_markers, adata.n_vars),
        pts=True,
    )
    ranked = sc.get.rank_genes_groups_df(adata, group=None)
    ranked = ranked[ranked["names"].notna()].copy()
    ranked["case_id"] = case_id
    ranked["cluster_id"] = ranked["group"].astype(str)
    ranked["rank"] = ranked.groupby(["case_id", "cluster_id"]).cumcount()
    ranked = ranked.rename(columns={"names": "gene"})

    marker_rows: list[dict[str, Any]] = []
    for row in ranked.itertuples(index=False):
        record = {
            "case_id": case_id,
            "cluster_id": str(getattr(row, "cluster_id")),
            "rank": int(getattr(row, "rank")),
            "gene": str(getattr(row, "gene")),
            "score": _json_float(getattr(row, "scores", None)),
            "logfoldchange": _json_float(getattr(row, "logfoldchanges", None)),
            "pval_adj": _json_float(getattr(row, "pvals_adj", None)),
            "pct_in_cluster": _json_float(getattr(row, "pct_nz_group", None)),
            "pct_out_cluster": _json_float(getattr(row, "pct_nz_reference", None)),
        }
        marker_rows.append(record)

    summary_rows: list[dict[str, Any]] = []
    for cluster_id, count in cluster_counts.items():
        genes = [
            row["gene"]
            for row in marker_rows
            if row["cluster_id"] == str(cluster_id) and row["rank"] < min(10, top_markers)
        ]
        summary_rows.append(
            {
                "case_id": case_id,
                "cluster_id": str(cluster_id),
                "n_cells": int(count),
                "top_marker_genes": genes,
            }
        )

    run_meta = {
        "case_id": case_id,
        "h5ad_path": str(path),
        "raw_shape": raw_shape,
        "sampled_n_obs_before_qc": sampled_n_obs,
        "n_obs_after_qc": int(adata.n_obs),
        "n_vars_post_hvg": int(adata.n_vars),
        "subsample_cells": int(subsample_cells),
        "n_top_genes": int(n_top_genes),
        "top_markers": int(top_markers),
        "random_state": int(random_state),
        "min_counts_per_cell": int(min_counts_per_cell),
        "min_cells_per_gene": int(min_cells_per_gene),
        "hvg_flavor": hvg_flavor,
        "cluster_key": "_ct_leiden",
        "cluster_resolution": 0.8,
        "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
        "training_called": False,
        "h5ad_written": False,
    }
    return marker_rows, summary_rows, run_meta


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def _write_markdown(summary_rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Cluster marker audit",
        "",
        "Marker genes were extracted without model retraining or h5ad writes.",
        "Clusters are Leiden `_ct_leiden` groups from the no-training preprocessing path.",
        "",
    ]
    for case_id in DEFAULT_CASES:
        case_rows = [row for row in summary_rows if row["case_id"] == case_id]
        if not case_rows:
            continue
        lines.extend([f"## {case_id}", "", "| Cluster | n cells | Top marker genes |", "| --- | ---: | --- |"])
        for row in case_rows:
            genes = ", ".join(row["top_marker_genes"])
            lines.append(f"| {row['cluster_id']} | {row['n_cells']} | {genes} |")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated case IDs.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--subsample-cells", type=int, default=3000)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--top-markers", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-counts-per-cell", type=int, default=200)
    parser.add_argument("--min-cells-per-gene", type=int, default=10)
    args = parser.parse_args()

    case_ids = _parse_cases(args.cases)
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_markers: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for case_id in case_ids:
        print(f"[markers] {case_id}", flush=True)
        marker_rows, summary_rows, run_meta = _extract_case_markers(
            case_id,
            subsample_cells=args.subsample_cells,
            n_top_genes=args.n_top_genes,
            top_markers=args.top_markers,
            random_state=args.random_state,
            min_counts_per_cell=args.min_counts_per_cell,
            min_cells_per_gene=args.min_cells_per_gene,
        )
        all_markers.extend(marker_rows)
        all_summary.extend(summary_rows)
        metadata.append(run_meta)

    markers_csv = args.outdir / "cluster_markers.csv"
    summary_csv = args.outdir / "cluster_marker_summary.csv"
    metadata_json = args.outdir / "cluster_marker_run_metadata.json"
    markdown = args.outdir / "cluster_marker_audit.md"

    pd.DataFrame(all_markers).to_csv(markers_csv, index=False)
    pd.DataFrame(all_summary).to_csv(summary_csv, index=False)
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_markdown(all_summary, markdown)

    print(f"wrote {markers_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {metadata_json}")
    print(f"wrote {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
