#!/usr/bin/env python3
"""build_datasets_csv.py — Build source maps and the canonical datasets.csv.

Two modes (run one at a time):

  --scan-host
      Walk pre-known host directories, find .h5ad files, write:
        data/scrna_source_map.csv   (filename_key -> absolute_source_path)
        data/scatac_source_map.csv  (filename_key -> absolute_source_path)
      Best-effort modality detection: scATAC if filename contains 'atac', 'ATA',
      or filename_key starts with 'ATA_'; else scRNA.

  --build-canonical
      Read workspace/data/scrna/ and workspace/data/scatac/ (populated by
      symlinks + any fetched h5ad files). Parse each h5ad for AnnData metadata.
      Write scccvgben/data/datasets.csv with schema:
        filename_key, GSE, tissue, species, category, description, modality,
        cell_count, drop_status

Usage:
    python scripts/build_datasets_csv.py --scan-host
    python scripts/build_datasets_csv.py --build-canonical
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATASETS_CSV = REPO_ROOT / "scccvgben" / "data" / "datasets.csv"
SCRNA_DIR = REPO_ROOT / "workspace" / "data" / "scrna"
SCATAC_DIR = REPO_ROOT / "workspace" / "data" / "scatac"
DROPPED_CSV = DATA_DIR / "dropped_scatac_v2.csv"

# Host directories to scan
SCAN_DIRS = [
    Path("/home/zeyufu/LAB/DATA"),
    Path("/home/zeyufu/LAB/SCRL"),
    Path("/home/zeyufu/vGAE_LAB/data"),
    Path("/home/zeyufu/Downloads/DevelopmentDatasets"),
    Path("/home/zeyufu/LAB/MCC_results"),
    Path("/home/zeyufu/Desktop/HMJ"),
    Path("/home/zeyufu/Desktop/CSD"),
    Path("/home/zeyufu/Desktop/WC"),
    Path("/home/zeyufu/Desktop/LSK-LCN-Publicdata"),
]

ATAC_PATTERNS = re.compile(r"atac|ATA_|ATAC", re.IGNORECASE)


def _is_scatac(path: Path) -> bool:
    return bool(ATAC_PATTERNS.search(path.name))


def _parse_gse(name: str) -> str:
    m = re.search(r"GSE\d+|GSM\d+", name)
    return m.group(0) if m else ""


def _infer_tissue(desc: str) -> str:
    desc_lower = desc.lower()
    tissue_map = {
        "pbmc": "PBMC", "blood": "blood", "lung": "lung", "brain": "brain",
        "liver": "liver", "kidney": "kidney", "heart": "heart",
        "bone": "bone_marrow", "marrow": "bone_marrow", "tumor": "tumor",
        "cancer": "tumor", "skin": "skin", "breast": "breast",
        "colon": "colon", "pancrea": "pancreas", "retina": "retina",
        "thymus": "thymus", "spleen": "spleen", "muscle": "muscle",
        "adipos": "adipose", "ovari": "ovary", "prostat": "prostate",
        "psc": "stem_cell", "esc": "stem_cell", "pdx": "PDX",
    }
    for kw, tissue in tissue_map.items():
        if kw in desc_lower:
            return tissue
    return "other"


def _infer_species(desc: str) -> str:
    if any(kw in desc for kw in ("Mm", "Mouse", "mouse", "murine")):
        return "mouse"
    return "human"


def _parse_filename_key(stem: str) -> dict:
    """Parse category_GSE_description from stem (strips _df suffix)."""
    name = re.sub(r"_df$", "", stem)
    m = re.match(r"^([A-Za-z]+)_(GSE\d+|GSM\d+)_(.+)$", name)
    if m:
        return {
            "category": m.group(1),
            "GSE": m.group(2),
            "description": m.group(3),
        }
    return {"category": "Unknown", "GSE": _parse_gse(name), "description": name}


# ── mode 1: scan host ────────────────────────────────────────────────────────

def cmd_scan_host() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    found: list[dict] = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            log.debug("Scan dir not found, skipping: %s", scan_dir)
            continue
        for h5ad in scan_dir.rglob("*.h5ad"):
            modality = "scatac" if _is_scatac(h5ad) else "scrna"
            stem = h5ad.stem
            found.append({
                "filename_key": stem,
                "absolute_source_path": str(h5ad.resolve()),
                "modality": modality,
            })

    log.info("Found %d .h5ad files across scan directories.", len(found))

    scrna_rows = [r for r in found if r["modality"] == "scrna"]
    scatac_rows = [r for r in found if r["modality"] == "scatac"]

    for rows, out_name in [
        (scrna_rows, "scrna_source_map.csv"),
        (scatac_rows, "scatac_source_map.csv"),
    ]:
        out_path = DATA_DIR / out_name
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["filename_key", "absolute_source_path"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"filename_key": row["filename_key"],
                                 "absolute_source_path": row["absolute_source_path"]})
        log.info("Wrote %d entries -> %s", len(rows), out_path)

    print(f"Scan complete: {len(scrna_rows)} scRNA, {len(scatac_rows)} scATAC h5ad files found.")


# ── mode 2: build canonical datasets.csv ────────────────────────────────────

def _load_dropped_keys() -> set[str]:
    if not DROPPED_CSV.exists():
        return set()
    with open(DROPPED_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        return {row["filename_key"] for row in reader}


def _cell_count(h5ad_path: Path) -> int:
    try:
        import anndata as ad
        adata = ad.read_h5ad(h5ad_path, backed="r")
        n = adata.n_obs
        adata.file.close()
        return n
    except Exception:
        return -1


def cmd_build_canonical() -> None:
    DATASETS_CSV.parent.mkdir(parents=True, exist_ok=True)
    dropped_keys = _load_dropped_keys()

    rows: list[dict] = []

    for modality, directory in [("scrna", SCRNA_DIR), ("scatac", SCATAC_DIR)]:
        if not directory.exists():
            log.warning("Directory not found: %s", directory)
            continue
        for h5ad in sorted(directory.iterdir()):
            if not (h5ad.name.endswith(".h5ad")):
                continue
            stem = h5ad.stem
            parsed = _parse_filename_key(stem)
            tissue = _infer_tissue(parsed["description"])
            species = _infer_species(parsed["description"])
            drop_status = "dropped" if stem in dropped_keys else "kept"

            # Try to get cell count (may fail for broken symlinks)
            cell_count = _cell_count(h5ad) if (h5ad.exists() and h5ad.stat().st_size > 0) else -1

            rows.append({
                "filename_key": stem,
                "GSE": parsed["GSE"],
                "tissue": tissue,
                "species": species,
                "category": parsed["category"],
                "description": parsed["description"],
                "modality": modality,
                "cell_count": cell_count if cell_count >= 0 else "",
                "drop_status": drop_status,
            })

    fieldnames = [
        "filename_key", "GSE", "tissue", "species", "category",
        "description", "modality", "cell_count", "drop_status",
    ]
    with open(DATASETS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d rows -> %s", len(rows), DATASETS_CSV)
    kept = sum(1 for r in rows if r["drop_status"] == "kept")
    print(f"datasets.csv: {len(rows)} total, {kept} kept ({DATASETS_CSV})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan-host", action="store_true",
                       help="Scan host directories and write source maps.")
    group.add_argument("--build-canonical", action="store_true",
                       help="Build scccvgben/data/datasets.csv from workspace h5ad files.")
    args = parser.parse_args()

    if args.scan_host:
        cmd_scan_host()
    elif args.build_canonical:
        cmd_build_canonical()


if __name__ == "__main__":
    main()
