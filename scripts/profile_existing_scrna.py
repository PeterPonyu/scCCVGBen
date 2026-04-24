#!/usr/bin/env python3
"""profile_existing_scrna.py — Profile the 55 existing scRNA baseline CSVs.

Reads filenames from CG_dl_merged/ (each named Can_GSE*_*_df.csv) and parses:
  - GSE accession
  - tissue / condition description (from filename)
  - category (Can, Hem, etc.)
  - organism (heuristic from description keywords)
  - cell_count (from the corresponding h5ad if locatable via scrna_source_map.csv)

Writes data/existing_scrna_diversity.csv, used by fetch_geo_scrna.py to bias
toward tissue/organism gaps when selecting the 45 new scRNA datasets.

Usage:
    python scripts/profile_existing_scrna.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CG_DL_MERGED = Path("/home/zeyufu/LAB/CCVGAE/CG_results/CG_dl_merged")
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SOURCE_MAP_PATH = DATA_DIR / "scrna_source_map.csv"


# ── organism heuristics ───────────────────────────────────────────────────────
_HUMAN_KEYWORDS = {"Hm", "human", "PBMC", "pbmc", "Tumor", "Cancer", "Lung",
                   "Brain", "Blood", "Liver", "Kidney", "Heart", "MCC", "bcc",
                   "scc", "Adre", "Myeloma", "AML", "CLL", "NHL", "DLBCL"}
_MOUSE_KEYWORDS = {"Mouse", "mouse", "Mm", "murine", "Mus"}


def _infer_organism(desc: str) -> str:
    for kw in _MOUSE_KEYWORDS:
        if kw in desc:
            return "mouse"
    for kw in _HUMAN_KEYWORDS:
        if kw in desc:
            return "human"
    return "unknown"


def _infer_tissue(desc: str) -> str:
    """Very rough tissue extraction from the description segment of the filename."""
    desc_lower = desc.lower()
    tissue_map = {
        "pbmc": "PBMC", "blood": "blood", "lung": "lung", "brain": "brain",
        "liver": "liver", "kidney": "kidney", "heart": "heart",
        "bone": "bone_marrow", "marrow": "bone_marrow", "tumor": "tumor",
        "cancer": "tumor", "skin": "skin", "breast": "breast",
        "colon": "colon", "pancrea": "pancreas", "retina": "retina",
        "thymus": "thymus", "spleen": "spleen", "muscle": "muscle",
        "adipos": "adipose", "ovari": "ovary", "prostat": "prostate",
    }
    for kw, tissue in tissue_map.items():
        if kw in desc_lower:
            return tissue
    return "other"


def _load_source_map() -> dict[str, Path]:
    """Load filename_key -> absolute_path from data/scrna_source_map.csv if present."""
    source_map: dict[str, Path] = {}
    if not SOURCE_MAP_PATH.exists():
        return source_map
    with open(SOURCE_MAP_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            source_map[row["filename_key"]] = Path(row["absolute_source_path"])
    return source_map


def _cell_count_from_h5ad(h5ad_path: Path) -> int | None:
    try:
        import anndata as ad
        adata = ad.read_h5ad(h5ad_path, backed="r")
        n = adata.n_obs
        adata.file.close()
        return n
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(DATA_DIR / "existing_scrna_diversity.csv"),
        help="Output CSV path."
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csvs = sorted(CG_DL_MERGED.glob("Can_*.csv"))
    log.info("Found %d CSVs in %s", len(csvs), CG_DL_MERGED)

    source_map = _load_source_map()
    log.info("Loaded %d h5ad source paths.", len(source_map))

    rows: list[dict] = []
    for csv_path in csvs:
        stem = csv_path.stem  # e.g. Can_GSE117988_MCCPBMCCancer_df
        # Strip trailing _df
        name = re.sub(r"_df$", "", stem)
        # Parse: Category_GSE_Description
        match = re.match(r"^([A-Za-z]+)_(GSE\d+)_(.+)$", name)
        if not match:
            log.warning("Unexpected filename pattern: %s", stem)
            category, gse, desc = "Unknown", "unknown", name
        else:
            category, gse, desc = match.group(1), match.group(2), match.group(3)

        organism = _infer_organism(desc)
        tissue = _infer_tissue(desc)

        # Try to get cell count
        cell_count = None
        filename_key = f"{category}_{gse}_{desc}"
        if filename_key in source_map:
            cell_count = _cell_count_from_h5ad(source_map[filename_key])

        rows.append({
            "filename_key": filename_key,
            "GSE": gse,
            "tissue": tissue,
            "category": category,
            "organism": organism,
            "description": desc,
            "cell_count": cell_count if cell_count is not None else "",
        })

    # Write output
    fieldnames = ["filename_key", "GSE", "tissue", "category", "organism", "description", "cell_count"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d rows to %s", len(rows), out_path)

    # Print tissue summary for quick review
    from collections import Counter
    tissue_counts = Counter(r["tissue"] for r in rows)
    print(f"\nExisting scRNA diversity ({len(rows)} datasets):")
    for tissue, count in tissue_counts.most_common():
        print(f"  {tissue}: {count}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
