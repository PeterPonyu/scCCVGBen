#!/usr/bin/env python3
"""reconcile_result_schema.py — Produce scCCVGBen-format per-dataset result CSVs.

The downstream integration target is the scCCVGBen reused-results layout
(legacy label-agreement columns dropped 2026-04-25):

  CG_dl_merged/{Category}_{GSE}_{desc}_df.csv      — 25 cols, methods as rows
                                                      header: method, ASW, ..., interpretation_intrin
                                                      rows:   PCA, KPCA, ..., scCCVGBen

  CG_atacs/tables/ATA_{GSE}_{desc}_df.csv          — 25 cols, methods as rows
                                                      header: "", ASW, ..., interpretation_intrin
                                                      rows:   LSI, PeakVI, PoissonVI, scCCVGBen
                                                      (method name is the pandas index, no 'method' column)

This script aggregates all axis outputs + reused CSVs and emits one canonical
per-dataset file in this exact format. File per file, drop-in compatible with
the reused CSVs for concat / figure generation.

Inputs (default):
    results/encoder_sweep/*.csv            (Axis A new rows; method = "scCCVGBen_{encoder}")
    results/graph_sweep/*.csv              (Axis B new rows; method = "scCCVGBen_GAT_{graph}")
    results/baselines/{scrna,scatac}_*.csv (Axis C new rows; method = baseline name)
    workspace/reused_results/scrna_baselines/*.csv   (reused scRNA)
    workspace/reused_results/axisA_GAT_scrna/*.csv   (same, GAT row consumed here)
    workspace/reused_results/scatac_baselines/*.csv  (reused scATAC)

Outputs:
    results/reconciled/scrna/{Category}_{GSE}_{desc}_df.csv
    results/reconciled/scatac/ATA_{GSE}_{desc}_df.csv

Filename resolution:
    If the dataset_key matches an existing CG_dl_merged / CG_atacs/tables
    filename, reuse that filename verbatim. Otherwise, consult
    scccvgben/data/datasets.csv (category + GSE + description) to build one.

Usage:
    python scripts/reconcile_result_schema.py
    python scripts/reconcile_result_schema.py --dry-run
    python scripts/reconcile_result_schema.py --in <dir>... --out results/reconciled/
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# scRNA canonical schema (matches CG_dl_merged; legacy label-agreement fields and COR removed 2026-04-25)
SCRNA_COLS = [
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

# scATAC canonical schema (matches CG_atacs/tables: method is the row-index)
SCATAC_COLS_INDEXED = [
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


def _classify(path: Path) -> tuple[str, str]:
    """Return (axis, modality) guessed from the path."""
    parts = [p.name for p in path.parents]
    name = path.name.lower()
    if "encoder_sweep" in parts or "axisA_GAT_scrna" in parts:
        axis = "A"
    elif "graph_sweep" in parts:
        axis = "B"
    else:
        axis = "C"
    if "scatac" in name or "atac" in name or name.startswith("ata_"):
        modality = "scatac"
    elif "scrna" in name or "_scrna" in name or name.startswith("scrna_"):
        modality = "scrna"
    elif any("scatac" in p.lower() or "atac" in p.lower() or "ATA" in p
             for p in [pp.name for pp in path.parents]):
        modality = "scatac"
    else:
        modality = "scrna"
    return axis, modality


def _strip_wrapper_prefixes(stem: str) -> str:
    """Normalise filename stem to scCCVGBen-style key.

    Removes wrappers that the new runners add:
      scrna_{key}     -> {key}
      scatac_{key}    -> {key}
      {key}_df        -> {key}
    """
    for pfx in ("scrna_", "scatac_"):
        if stem.startswith(pfx):
            stem = stem[len(pfx):]
            break
    if stem.endswith("_df"):
        stem = stem[:-3]
    return stem


def _infer_category(key: str, description: str) -> str:
    """Derive scCCVGBen-style category prefix for new scRNA datasets.

    Matches the patterns already in CG_dl_merged filenames:
      Can_  — cancer; Dev_  — development; sup_  — supplement; etc.
    """
    lo = (key + " " + str(description)).lower()
    if "cancer" in lo or "tumor" in lo or "mcc" in lo or "bcc" in lo or "mm" in lo:
        return "Can"
    if "dev" in lo or "embryo" in lo or "differentiation" in lo or "esc" in lo:
        return "Dev"
    if "sup" in lo or "wtko" in lo or "irall" in lo:
        return "sup"
    return "Gen"  # generic


def _lookup_scccvgben_filename(
    dataset_key: str,
    modality: str,
    manifest: pd.DataFrame | None,
    reuse_scrna_names: set[str],
    reuse_scatac_names: set[str],
) -> str:
    """Return the canonical {...}_df.csv filename for a given dataset_key.

    Strategy:
      1. If a reused file already exists for this key, use its exact name.
      2. Otherwise, infer category (Can/Dev/sup/Gen) + preserve GSE + desc,
         and emit {Category}_{rest_of_key}_df.csv (scRNA) or ATA_{key}_df.csv (scATAC).
    """
    key = _strip_wrapper_prefixes(dataset_key)

    # Direct name match to reused set (either modality)
    for pfx in ("",):
        candidate = f"{pfx}{key}_df.csv"
        if candidate in reuse_scrna_names or candidate in reuse_scatac_names:
            return candidate

    if modality == "scatac":
        # ATA_ prefix convention
        if not key.startswith("ATA_"):
            return f"ATA_{key}_df.csv"
        return f"{key}_df.csv"

    # scRNA: needs category prefix
    if re.match(r"^(Can|Dev|sup|Gen)_", key):
        return f"{key}_df.csv"

    desc = ""
    if manifest is not None:
        match = manifest[manifest["dataset_id"].astype(str) == key]
        if len(match):
            desc = str(match.iloc[0].get("notes", ""))
    cat = _infer_category(key, desc)
    return f"{cat}_{key}_df.csv"


def _normalise_scrna(df: pd.DataFrame) -> pd.DataFrame:
    """Return (n_rows, len(SCRNA_COLS)) DataFrame with canonical columns.

    Drops Axis-A auxiliary columns (encoder, family, graph_method, dataset,
    Unnamed: 0). Ensures 'method' is populated; falls back to row index.
    """
    if "Unnamed: 0" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "method"})
    for aux in ("encoder", "family", "graph_method", "dataset"):
        if aux in df.columns:
            df = df.drop(columns=[aux])
    for col in SCRNA_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[SCRNA_COLS].copy()


def _normalise_scatac(df: pd.DataFrame) -> pd.DataFrame:
    """scATAC output format: method as row-index, 26 metric columns."""
    if "Unnamed: 0" in df.columns and "method" not in df.columns:
        df = df.rename(columns={"Unnamed: 0": "method"})
    for aux in ("encoder", "family", "graph_method", "dataset"):
        if aux in df.columns:
            df = df.drop(columns=[aux])
    if "method" not in df.columns:
        df["method"] = "UNKNOWN"
    for col in SCATAC_COLS_INDEXED:
        if col not in df.columns:
            df[col] = pd.NA
    out = df.set_index("method")[SCATAC_COLS_INDEXED].copy()
    out.index.name = None  # CG_atacs/tables leaves index column unnamed
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="in_dirs", nargs="+", default=[
            str(REPO_ROOT / "results" / "encoder_sweep"),
            str(REPO_ROOT / "results" / "graph_sweep"),
            str(REPO_ROOT / "results" / "baselines"),
            str(REPO_ROOT / "workspace" / "reused_results"),
        ],
        help="Input directories to scan recursively for *.csv",
    )
    parser.add_argument(
        "--out", dest="out_dir",
        default=str(REPO_ROOT / "results" / "reconciled"),
        help="Output root (will get scrna/ + scatac/ subdirs).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_scrna = out_dir / "scrna"
    out_scatac = out_dir / "scatac"
    if not args.dry_run:
        out_scrna.mkdir(parents=True, exist_ok=True)
        out_scatac.mkdir(parents=True, exist_ok=True)

    # Manifest for filename resolution + category inference
    manifest_path = REPO_ROOT / "data" / "revised_methodology_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else None

    reuse_scrna_dir = REPO_ROOT / "workspace" / "reused_results" / "scrna_baselines"
    reuse_scatac_dir = REPO_ROOT / "workspace" / "reused_results" / "scatac_baselines"
    reuse_scrna_names = {p.name for p in reuse_scrna_dir.glob("*.csv")} if reuse_scrna_dir.exists() else set()
    reuse_scatac_names = {p.name for p in reuse_scatac_dir.glob("*.csv")} if reuse_scatac_dir.exists() else set()

    # Bucket rows by canonical filename
    scrna_buckets: dict[str, list[pd.DataFrame]] = {}
    scatac_buckets: dict[str, list[pd.DataFrame]] = {}
    total_files = 0

    for d in args.in_dirs:
        p = Path(d)
        if not p.exists():
            log.warning("Input dir missing: %s", p)
            continue
        for csv in p.rglob("*.csv"):
            try:
                df = pd.read_csv(csv)
                if len(df) == 0:
                    continue
                axis, modality = _classify(csv)
                dataset_key = _strip_wrapper_prefixes(csv.stem)
                canonical_name = _lookup_scccvgben_filename(
                    dataset_key, modality, manifest,
                    reuse_scrna_names, reuse_scatac_names,
                )
                if modality == "scrna":
                    scrna_buckets.setdefault(canonical_name, []).append(_normalise_scrna(df))
                else:
                    scatac_buckets.setdefault(canonical_name, []).append(_normalise_scatac(df))
                total_files += 1
            except Exception as exc:
                log.error("Skipping %s: %s", csv, exc)

    log.info(
        "Scanned %d files -> %d scRNA canonical datasets + %d scATAC canonical datasets",
        total_files, len(scrna_buckets), len(scatac_buckets),
    )

    # Write per-dataset merged CSVs
    n_scrna_written = 0
    n_scatac_written = 0
    for name, frames in scrna_buckets.items():
        merged = pd.concat(frames, ignore_index=True)
        # De-duplicate by method (keep last → favours our fresh runs over reused)
        merged = merged.drop_duplicates(subset="method", keep="last")
        if args.dry_run:
            continue
        merged.to_csv(out_scrna / name, index=False)
        n_scrna_written += 1
    for name, frames in scatac_buckets.items():
        merged = pd.concat(frames, axis=0)
        merged = merged[~merged.index.duplicated(keep="last")]
        if args.dry_run:
            continue
        merged.to_csv(out_scatac / name, index=True)
        n_scatac_written += 1

    print(
        f"\nReconcile summary:\n"
        f"  scanned:     {total_files} raw CSVs\n"
        f"  scRNA out:   {n_scrna_written} files -> {out_scrna}\n"
        f"  scATAC out:  {n_scatac_written} files -> {out_scatac}"
    )
    if args.dry_run:
        print("  (dry-run; no files written)")


if __name__ == "__main__":
    main()
