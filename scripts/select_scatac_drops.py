#!/usr/bin/env python3
"""select_scatac_drops.py — Select 15 scATAC datasets to drop from the 115.

Drop criteria (spec Round 2):
  1. Lowest cell count (ascending).
  2. Duplicate tissue (prefer keeping unique tissues).

Reads cell counts from corresponding h5ad symlinks in workspace/data/scatac/
if available; otherwise marks as unknown (treated as large, not dropped for
cell-count criterion but may be dropped for tissue-duplication).

Writes data/dropped_scatac_v2.csv with exactly 15 rows:
  filename_key, GSE, tissue, cell_count, drop_reason

Usage:
    python scripts/select_scatac_drops.py [--n-drops 15]
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SCATAC_BASELINES_DIR = REPO_ROOT / "workspace" / "reused_results" / "scatac_baselines"
SCATAC_H5AD_DIR = REPO_ROOT / "workspace" / "data" / "scatac"
SCATAC_SOURCE = Path("/home/zeyufu/LAB/CCVGAE/CG_results/CG_atacs/tables")

_TISSUE_KEYWORDS = {
    "pbmc": "PBMC", "blood": "blood", "lung": "lung", "brain": "brain",
    "liver": "liver", "kidney": "kidney", "heart": "heart",
    "bone": "bone_marrow", "marrow": "bone_marrow", "tumor": "tumor",
    "cancer": "tumor", "skin": "skin", "breast": "breast",
    "colon": "colon", "pancrea": "pancreas", "retina": "retina",
    "thymus": "thymus", "spleen": "spleen", "muscle": "muscle",
    "adipos": "adipose", "ovari": "ovary", "prostat": "prostate",
    "atac": "unspecified", "psc": "stem_cell", "esc": "stem_cell",
    "pdx": "PDX", "fibroblast": "fibroblast", "epithe": "epithelium",
}


def _infer_tissue(name: str) -> str:
    name_lower = name.lower()
    for kw, tissue in _TISSUE_KEYWORDS.items():
        if kw in name_lower:
            return tissue
    return "other"


def _parse_gse(name: str) -> str:
    m = re.search(r"GSE\d+|GSM\d+", name)
    return m.group(0) if m else "unknown"


def _cell_count_from_h5ad(filename_key: str) -> int | None:
    """Try to read cell count from h5ad in workspace/data/scatac/."""
    for suffix in ("", "_df"):
        h5ad = SCATAC_H5AD_DIR / f"{filename_key}{suffix}.h5ad"
        if h5ad.exists() or h5ad.is_symlink():
            try:
                import anndata as ad
                adata = ad.read_h5ad(h5ad, backed="r")
                n = adata.n_obs
                adata.file.close()
                return n
            except Exception:
                pass
    return None


def _collect_dataset_info(csv_files: list[Path]) -> list[dict]:
    """Build info dicts for each scATAC CSV file."""
    rows = []
    for f in csv_files:
        stem = f.stem  # e.g. ATA_GSE198730_aPSM_scATAC_rep1_filtered_peak_bc_matrix_df
        name = re.sub(r"_df$", "", stem)
        gse = _parse_gse(name)
        tissue = _infer_tissue(name)
        cell_count = _cell_count_from_h5ad(stem)
        rows.append({
            "filename_key": stem,
            "GSE": gse,
            "tissue": tissue,
            "cell_count": cell_count,
        })
    return rows


def _select_drops(rows: list[dict], n_drops: int = 15) -> list[dict]:
    """Select n_drops rows to drop by: low cell count + duplicate tissue."""
    # Separate known and unknown cell counts
    known = [r for r in rows if r["cell_count"] is not None]
    unknown = [r for r in rows if r["cell_count"] is None]

    # Sort known by ascending cell count
    known_sorted = sorted(known, key=lambda r: r["cell_count"])

    drops: list[dict] = []
    kept_tissues: Counter = Counter()

    # Pass 1: drop lowest cell-count datasets
    for row in known_sorted:
        if len(drops) >= n_drops:
            break
        drops.append({**row, "drop_reason": "low_cell_count"})

    # Pass 2: if still need more drops, drop duplicate tissues from unknown
    if len(drops) < n_drops:
        # Count tissues among known (non-dropped)
        remaining = [r for r in rows if r not in drops]
        tissue_seen: set[str] = set()
        for row in remaining:
            if row["tissue"] in tissue_seen and len(drops) < n_drops:
                drops.append({**row, "drop_reason": "duplicate_tissue"})
            tissue_seen.add(row["tissue"])

    return drops[:n_drops]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-drops", type=int, default=15,
                        help="Number of datasets to drop (default: 15).")
    parser.add_argument("--out", default=str(DATA_DIR / "dropped_scatac_v2.csv"),
                        help="Output CSV path.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer symlinked baselines dir; fall back to source
    search_dirs = [SCATAC_BASELINES_DIR, SCATAC_SOURCE]
    csv_files: list[Path] = []
    for d in search_dirs:
        if d.exists():
            csv_files = sorted(d.glob("*.csv"))
            if csv_files:
                log.info("Reading %d scATAC CSVs from %s", len(csv_files), d)
                break

    if not csv_files:
        log.error("No scATAC CSVs found. Run setup_symlinks.sh first.")
        raise SystemExit(1)

    rows = _collect_dataset_info(csv_files)
    log.info("Collected info for %d scATAC datasets.", len(rows))

    drops = _select_drops(rows, n_drops=args.n_drops)
    log.info("Selected %d datasets to drop.", len(drops))

    fieldnames = ["filename_key", "GSE", "tissue", "cell_count", "drop_reason"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for drop in drops:
            writer.writerow({k: drop.get(k, "") for k in fieldnames})

    log.info("Wrote %s", out_path)

    kept = len(rows) - len(drops)
    log.info("Kept: %d / %d scATAC datasets.", kept, len(rows))

    print(f"\nDropped {len(drops)} scATAC datasets -> {out_path}")
    for d in drops:
        print(f"  {d['filename_key']}  cells={d['cell_count']}  reason={d['drop_reason']}")


if __name__ == "__main__":
    main()
