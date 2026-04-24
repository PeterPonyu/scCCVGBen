#!/usr/bin/env python3
"""
build_inventory.py
Scans all known h5ad/h5 data locations, builds scRNA and scATAC inventories,
and writes summary markdown.
"""

import os
import re
import csv
import time
import signal
from pathlib import Path
from collections import defaultdict

# ── output paths ──────────────────────────────────────────────────────────────
OUT_DIR = Path("/home/zeyufu/LAB/scCCVGBen/data")
SCRNA_CSV  = OUT_DIR / "existing_scrna_diversity.csv"
SCATAC_CSV = OUT_DIR / "existing_scatac_inventory.csv"
SUMMARY_MD = OUT_DIR / "inventory_summary.md"

SCHEMA = ["dataset_id","gse","path_abs","tissue","organism",
          "cell_count","n_genes","modality","provenance_dir","format","priority_guess"]

# ── helpers ───────────────────────────────────────────────────────────────────
SIZE_LIMIT = 5 * 1024**3   # 5 GB

def extract_gse(name: str) -> str:
    m = re.search(r'(GSE\d+)', name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'(GSM\d+)', name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "unknown"

def infer_organism(name: str) -> str:
    nl = name.lower()
    if re.search(r'_hm|human|hesc|pbmc|breast|gastric|melanoma|nsclc|lym|lymph', nl):
        return "human"
    if re.search(r'_mm|mouse|murine|lsk|myo|paul|dentate|spinoid|bleo|zebrafish', nl):
        return "mouse"
    if re.search(r'zebrafish', nl):
        return "zebrafish"
    return "unknown"

def infer_tissue(name: str) -> str:
    nl = name.lower()
    mapping = [
        (r'bm|bone.?marrow|hsc|lsk|hemato|irall|pbmc', "bone_marrow_blood"),
        (r'brain|dentate|neuron|cortex|spinal|spinoid', "brain_neural"),
        (r'lung', "lung"),
        (r'breast|bc(?:ec|stroma|ell)', "breast"),
        (r'liver|hepato', "liver"),
        (r'gastric|stomach', "stomach"),
        (r'muscle|myo|myoblast', "muscle"),
        (r'retina', "retina"),
        (r'thymus|tcell|cd8|treg|nk', "immune"),
        (r'hesc|esC|stem', "stem_cell"),
        (r'kidney|urine', "kidney"),
        (r'pituitary|pit', "pituitary"),
        (r'pancrea|endo(?:crin)?', "pancreas"),
        (r'skin|melanoma|scc|bcc', "skin"),
        (r'colon|colorect', "colon"),
        (r'teeth', "teeth"),
        (r'mono(?:cyte)?', "monocyte"),
        (r'prostate|vcap', "prostate"),
        (r'atac', "atac_generic"),
    ]
    for pattern, tissue in mapping:
        if re.search(pattern, nl):
            return tissue
    return "unknown"

def priority_guess(path_str: str, name: str) -> int:
    lower = (path_str + name).lower()
    if any(k in lower for k in ["cancer", "tumor", "development", "dev", "atac"]):
        return 1
    return 2

def timeout_handler(signum, frame):
    raise TimeoutError("read timeout")

def read_h5ad_meta(path: Path):
    """Returns (n_obs, n_vars, obs_columns) or raises."""
    import anndata as ad
    try:
        adata = ad.read_h5ad(str(path), backed='r')
        n_obs  = adata.n_obs
        n_vars = adata.n_vars
        cols   = list(adata.obs.columns)
        adata.file.close()
        return n_obs, n_vars, cols
    except Exception:
        adata = ad.read_h5ad(str(path))
        return adata.n_obs, adata.n_vars, list(adata.obs.columns)

def read_h5_10x_meta(path: Path):
    """Try reading 10x HDF5 (CellRanger output) for cell/feature counts."""
    import h5py
    try:
        with h5py.File(str(path), 'r') as f:
            # CellRanger v3+
            if 'matrix' in f:
                shape = f['matrix']['shape'][:]
                return int(shape[1]), int(shape[0])   # (n_obs, n_vars)
            # CellRanger v2 / snap ATAC
            for key in f.keys():
                if 'barcodes' in f[key]:
                    n_obs = len(f[key]['barcodes'])
                    n_vars = len(f[key].get('genes', f[key].get('features', f[key].get('peaks', []))))
                    return n_obs, n_vars
        return None, None
    except Exception:
        return None, None

# ── ATAC RAW directory detection ──────────────────────────────────────────────
ATAC_RAW_DIRS = [
    "/home/zeyufu/Downloads/ATAC_data/GSE190162_RAW",
    "/home/zeyufu/Downloads/ATAC_data/GSE192947_RAW",
    "/home/zeyufu/Downloads/ATAC_data/GSE199556_RAW",
    "/home/zeyufu/Downloads/ATAC_data/GSE211155_RAW",
    "/home/zeyufu/Downloads/ATAC_data/GSE225803_RAW",
    "/home/zeyufu/Downloads/ATAC_data/GSE266511_RAW",
    "/home/zeyufu/Downloads/ATAC_data/GSE275786_RAW.",
    "/home/zeyufu/Downloads/ATAC_data/GSE284492_RAW",
    "/home/zeyufu/Downloads/ATAC_data/GSE292195_RAW",
    "/home/zeyufu/LAB/SCRL/data/GSE117498_RAW",
    "/home/zeyufu/LAB/SCRL/data/GSE137540_RAW",
]
SCRNA_RAW_DIRS = [
    "/home/zeyufu/LAB/SCRL/data/AML",
    "/home/zeyufu/LAB/SCRL/data/DentateGyrus",
    "/home/zeyufu/LAB/SCRL/data/Gastrulation",
    "/home/zeyufu/LAB/SCRL/data/moignard15",
    "/home/zeyufu/LAB/SCRL/data/paul15",
]

# ── file collections ──────────────────────────────────────────────────────────
# Tuples: (abs_path, provenance_dir, modality_hint)
# modality_hint: 'rna', 'atac', 'unknown'

SCRNA_FILES = []
SCATAC_FILES = []

def collect_h5ad_files(directory, modality_hint='rna', recursive=True):
    d = Path(directory)
    if not d.exists():
        return []
    pattern = '**/*.h5ad' if recursive else '*.h5ad'
    return [(p, str(d), modality_hint) for p in d.glob(pattern)]

def collect_h5_files(directory, modality_hint='atac', recursive=True):
    d = Path(directory)
    if not d.exists():
        return []
    pattern = '**/*.h5' if recursive else '*.h5'
    return [(p, str(d), modality_hint) for p in d.glob(pattern)]

# ── ATAC h5ad sources ─────────────────────────────────────────────────────────
SCATAC_FILES += collect_h5ad_files("/home/zeyufu/Downloads/ATAC_data", 'atac', recursive=False)
# ATAC .h5 (10x) in raw subdirs
for rd in ["/home/zeyufu/Downloads/ATAC_data/GSE192947_RAW",
           "/home/zeyufu/Downloads/ATAC_data/GSE199556_RAW",
           "/home/zeyufu/Downloads/ATAC_data/GSE211155_RAW",
           "/home/zeyufu/Downloads/ATAC_data/GSE225803_RAW",
           "/home/zeyufu/Downloads/ATAC_data/GSE266511_RAW",
           "/home/zeyufu/Downloads/ATAC_data/GSE284492_RAW",
           "/home/zeyufu/Downloads/ATAC_data/GSE292195_RAW",
           "/home/zeyufu/Downloads/ATAC_data/GSE206767_filtered_peak_bc_matrix",
           "/home/zeyufu/Downloads/ATAC_data/GSM5124061_PDX_390_vehicle_peak_bc_matrix",
           ]:
    SCATAC_FILES += collect_h5_files(rd, 'atac', recursive=False)

# 10x ATAC h5 in LAB/DATA
SCATAC_FILES += collect_h5_files("/home/zeyufu/LAB/DATA/10x_scATAC_h5", 'atac', recursive=False)
# processed_datasets (ATAC annotated)
SCATAC_FILES += collect_h5ad_files("/home/zeyufu/LAB/processed_datasets", 'atac', recursive=False)
# IAODEVAE atac output
SCATAC_FILES += [
    (Path("/home/zeyufu/LAB/IAODEVAE/iAODE/examples/outputs/atacseq_annotation/annotated_peaks.h5ad"),
     "/home/zeyufu/LAB/IAODEVAE", 'atac'),
]
# spatial (treat as rna)
SCATAC_FILES += collect_h5ad_files("/home/zeyufu/Downloads/Spatial_data", 'atac', recursive=False)

# ── scRNA h5ad sources ────────────────────────────────────────────────────────
for src in [
    "/home/zeyufu/Downloads/DevelopmentDatasets",
    "/home/zeyufu/Downloads/DevelopmentDatasets2",
    "/home/zeyufu/Downloads/CancerDatasets",
    "/home/zeyufu/Downloads/CancerDatasets2",
]:
    SCRNA_FILES += collect_h5ad_files(src, 'rna', recursive=False)

# LAB/DATA scRNA
for p in Path("/home/zeyufu/LAB/DATA").glob("*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/DATA", 'rna'))
# paul15.h5 in LAB/DATA - scRNA
SCRNA_FILES.append((Path("/home/zeyufu/LAB/DATA/paul15/paul15.h5"),
                    "/home/zeyufu/LAB/DATA/paul15", 'rna'))

# vGAE_LAB
for p in Path("/home/zeyufu/vGAE_LAB/data").glob("*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/vGAE_LAB/data", 'rna'))
SCRNA_FILES.append((Path("/home/zeyufu/vGAE_LAB/data/paul15/paul15.h5"),
                    "/home/zeyufu/vGAE_LAB/data/paul15", 'rna'))

# SCRL
for p in Path("/home/zeyufu/LAB/SCRL").glob("*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/SCRL", 'rna'))
for p in Path("/home/zeyufu/LAB/SCRL/data").glob("*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/SCRL/data", 'rna'))
SCRNA_FILES.append((Path("/home/zeyufu/LAB/SCRL/data/paul15/paul15.h5"),
                    "/home/zeyufu/LAB/SCRL/data/paul15", 'rna'))
SCRNA_FILES.append((Path("/home/zeyufu/LAB/SCRL/data/Gastrulation/erythroid_lineage.h5ad"),
                    "/home/zeyufu/LAB/SCRL/data/Gastrulation", 'rna'))

# Downloads root level
SCRNA_FILES.append((Path("/home/zeyufu/Downloads/adata.h5ad"),
                    "/home/zeyufu/Downloads", 'rna'))
SCRNA_FILES.append((Path("/home/zeyufu/Downloads/hESC_GSE144024.h5ad"),
                    "/home/zeyufu/Downloads", 'rna'))

# SCFOCUS export_datas
for p in Path("/home/zeyufu/LAB/SCFOCUS/export_datas").glob("*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/SCFOCUS/export_datas", 'rna'))
SCRNA_FILES.append((Path("/home/zeyufu/LAB/SCFOCUS/data/pbmc3k_raw.h5ad"),
                    "/home/zeyufu/LAB/SCFOCUS/data", 'rna'))

# IVAE
for p in Path("/home/zeyufu/LAB/IVAE").glob("*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/IVAE", 'rna'))

# IAODEVAE results (scRNA processed)
for p in Path("/home/zeyufu/LAB/IAODEVAE/iAODE_results").glob("**/*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/IAODEVAE/iAODE_results", 'rna'))

# IAODEVAE examples data
SCRNA_FILES.append((Path("/home/zeyufu/LAB/IAODEVAE/iAODE/examples/data/mouse_brain_5k_v1.1.h5"),
                    "/home/zeyufu/LAB/IAODEVAE/iAODE/examples/data", 'rna'))

# CCVGAE results
for p in Path("/home/zeyufu/LAB/CCVGAE/CG_results").glob("*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/CCVGAE/CG_results", 'rna'))

# MCC_results
for p in Path("/home/zeyufu/LAB/MCC_results").glob("*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/MCC_results", 'rna'))

# CODEVAE batch copies (skip model weights)
for p in Path("/home/zeyufu/LAB/CODEVAE").glob("*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/CODEVAE", 'rna'))
for p in Path("/home/zeyufu/LAB/CODEVAE/code_batch_2000").glob("*.h5ad"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/CODEVAE/code_batch_2000", 'rna'))

# LAIOR scRNA data (non-model-weights)
for p in Path("/home/zeyufu/LAB/LAIOR/Liora/laior/unified_models1/scDeepCluster-main/scRNA-seq data").glob("*.h5"):
    SCRNA_FILES.append((p, "/home/zeyufu/LAB/LAIOR/scRNA-seq_data", 'rna'))

# ── processing ────────────────────────────────────────────────────────────────
scrna_rows = []
scatac_rows = []
failed_files = []
skipped_large = []
seen_paths = set()

def make_row(path: Path, prov_dir: str, modality: str) -> dict | None:
    global failed_files, skipped_large
    abs_path = str(path.resolve())
    if abs_path in seen_paths:
        return None
    seen_paths.add(abs_path)

    if not path.exists():
        return None

    size = path.stat().st_size
    fname = path.name
    stem  = path.stem

    gse  = extract_gse(stem)
    org  = infer_organism(stem)
    tiss = infer_tissue(stem)
    prio = priority_guess(prov_dir, stem)
    fmt  = "h5ad" if path.suffix == ".h5ad" else "h5_10x"

    cell_count = "unknown"
    n_genes    = "unknown"

    # skip >5GB with timeout
    if size > SIZE_LIMIT:
        skipped_large.append(abs_path)
        cell_count = "SKIPPED_>5GB"
        n_genes    = "SKIPPED_>5GB"
    else:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)   # 2 min timeout per file
        try:
            if fmt == "h5ad":
                n_obs, n_vars, cols = read_h5ad_meta(path)
                cell_count = str(n_obs)
                n_genes    = str(n_vars)
                # try to refine org/tissue from obs columns
                for c in cols:
                    cl = c.lower()
                    if 'tissue' in cl or 'organ' in cl:
                        pass   # would need actual values – skip for speed
            else:
                n_obs, n_vars = read_h5_10x_meta(path)
                if n_obs is not None:
                    cell_count = str(n_obs)
                    n_genes    = str(n_vars)
        except TimeoutError:
            failed_files.append(f"TIMEOUT: {abs_path}")
            cell_count = "TIMEOUT"
            n_genes    = "TIMEOUT"
        except Exception as e:
            failed_files.append(f"ERROR ({e.__class__.__name__}): {abs_path}")
            cell_count = "ERROR"
            n_genes    = "ERROR"
        finally:
            signal.alarm(0)

    dataset_id = stem.replace(" ", "_")
    return {
        "dataset_id":    dataset_id,
        "gse":           gse,
        "path_abs":      abs_path,
        "tissue":        tiss,
        "organism":      org,
        "cell_count":    cell_count,
        "n_genes":       n_genes,
        "modality":      modality,
        "provenance_dir":prov_dir,
        "format":        fmt,
        "priority_guess":str(prio),
    }

print("Processing scRNA files ...")
for path, prov, hint in SCRNA_FILES:
    row = make_row(Path(path), prov, "scRNA")
    if row:
        scrna_rows.append(row)
        print(f"  [scRNA] {row['dataset_id']}  cells={row['cell_count']}  genes={row['n_genes']}")

print("\nProcessing scATAC files ...")
for path, prov, hint in SCATAC_FILES:
    row = make_row(Path(path), prov, "scATAC")
    if row:
        scatac_rows.append(row)
        print(f"  [scATAC] {row['dataset_id']}  cells={row['cell_count']}")

# ── RAW directory entries ─────────────────────────────────────────────────────
def add_raw_entry(raw_dir: str, modality: str, rows: list):
    d = Path(raw_dir)
    if not d.exists():
        return
    gse  = extract_gse(d.name)
    prio = 1 if any(k in raw_dir.lower() for k in ["cancer","atac","dev"]) else 2
    rows.append({
        "dataset_id":    d.name,
        "gse":           gse,
        "path_abs":      str(d.resolve()),
        "tissue":        infer_tissue(d.name),
        "organism":      infer_organism(d.name),
        "cell_count":    "unknown",
        "n_genes":       "unknown",
        "modality":      modality,
        "provenance_dir":str(d.parent),
        "format":        "raw_directory",
        "priority_guess":str(prio),
    })

for rd in ATAC_RAW_DIRS:
    add_raw_entry(rd, "scATAC", scatac_rows)
for rd in SCRNA_RAW_DIRS:
    add_raw_entry(rd, "scRNA", scrna_rows)

# Also GSE137540_RAW.tar in SCRL
scrna_rows.append({
    "dataset_id":    "GSE137540_RAW",
    "gse":           "GSE137540",
    "path_abs":      "/home/zeyufu/LAB/SCRL/data/GSE137540_RAW.tar",
    "tissue":        "unknown",
    "organism":      "unknown",
    "cell_count":    "unknown",
    "n_genes":       "unknown",
    "modality":      "scRNA",
    "provenance_dir":"/home/zeyufu/LAB/SCRL/data",
    "format":        "raw_tarball",
    "priority_guess":"2",
})

# ── write CSVs ────────────────────────────────────────────────────────────────
def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA)
        w.writeheader()
        w.writerows(rows)

write_csv(scrna_rows, SCRNA_CSV)
write_csv(scatac_rows, SCATAC_CSV)
print(f"\nWrote {len(scrna_rows)} scRNA rows -> {SCRNA_CSV}")
print(f"Wrote {len(scatac_rows)} scATAC rows -> {SCATAC_CSV}")

# ── compute summary stats ─────────────────────────────────────────────────────
def count_by(rows, field):
    d = defaultdict(int)
    for r in rows:
        d[r[field]] += 1
    return dict(sorted(d.items(), key=lambda x: -x[1]))

rna_tissue   = count_by(scrna_rows,  "tissue")
rna_org      = count_by(scrna_rows,  "organism")
atac_tissue  = count_by(scatac_rows, "tissue")
atac_org     = count_by(scatac_rows, "organism")

rna_prov  = count_by(scrna_rows,  "provenance_dir")
atac_prov = count_by(scatac_rows, "provenance_dir")

# GSE collision detection (same GSE across multiple provenance dirs, actual h5ad files only)
gse_dirs = defaultdict(set)
for r in scrna_rows + scatac_rows:
    if r["gse"] not in ("unknown",) and r["format"] in ("h5ad","h5_10x"):
        gse_dirs[r["gse"]].add(r["provenance_dir"])
gse_conflicts = {k: sorted(v) for k, v in gse_dirs.items() if len(v) > 1}

# ── write summary.md ──────────────────────────────────────────────────────────
def bar(d, top=15):
    lines = []
    total = sum(d.values())
    for k, v in list(d.items())[:top]:
        pct = 100*v/total if total else 0
        bar_str = "█" * int(pct/5)
        lines.append(f"| {k:<30} | {v:>5} | {pct:5.1f}% | {bar_str} |")
    return "\n".join(lines)

md = f"""# scCCVGBen Data Inventory Summary
Generated: 2026-04-23

## Overview
| Modality | Count |
|----------|-------|
| scRNA    | {len(scrna_rows):>5} |
| scATAC   | {len(scatac_rows):>5} |
| **Total**| **{len(scrna_rows)+len(scatac_rows)}** |

---

## scRNA Distribution

### By Tissue
| Tissue | Count | % | Bar |
|--------|------:|---|-----|
{bar(rna_tissue)}

### By Organism
| Organism | Count | % | Bar |
|----------|------:|---|-----|
{bar(rna_org)}

### Top Provenance Directories
| Directory | Count |
|-----------|------:|
""" + "\n".join(f"| {k} | {v} |" for k, v in list(rna_prov.items())[:10]) + f"""

---

## scATAC Distribution

### By Tissue
| Tissue | Count | % | Bar |
|--------|------:|---|-----|
{bar(atac_tissue)}

### By Organism
| Organism | Count | % | Bar |
|----------|------:|---|-----|
{bar(atac_org)}

### Top Provenance Directories
| Directory | Count |
|-----------|------:|
""" + "\n".join(f"| {k} | {v} |" for k, v in list(atac_prov.items())[:10]) + f"""

---

## Duplicate GSE Alerts
_Same GSE accession found in multiple provenance directories (h5ad/h5_10x files only):_

"""

if gse_conflicts:
    for gse, dirs in sorted(gse_conflicts.items()):
        md += f"- **{gse}**: {', '.join(dirs)}\n"
else:
    md += "_No GSE conflicts detected._\n"

md += f"""
---

## Large Files Skipped (>5 GB)
"""
if skipped_large:
    for f in skipped_large:
        md += f"- `{f}`\n"
else:
    md += "_None._\n"

md += f"""
---

## Files With Read Errors / Timeouts
"""
if failed_files:
    for f in failed_files:
        md += f"- `{f}`\n"
else:
    md += "_None._\n"

md += f"""
---

## RAW / Unexpanded Archives
_Directories or tarballs not yet converted to h5ad:_

### scATAC RAW Directories
"""
for rd in ATAC_RAW_DIRS:
    if Path(rd).exists():
        md += f"- `{rd}`\n"

md += "\n### scRNA RAW Directories / Tarballs\n"
for rd in SCRNA_RAW_DIRS + ["/home/zeyufu/LAB/SCRL/data/GSE137540_RAW.tar",
                              "/home/zeyufu/LAB/SCRL/data/GSE185834_RAW.tar",
                              "/home/zeyufu/LAB/SCRL/data/GSE185991_RAW.tar"]:
    if Path(rd).exists():
        md += f"- `{rd}`\n"

SUMMARY_MD.write_text(md)
print(f"Wrote summary -> {SUMMARY_MD}")
print(f"\nFailed/error files: {len(failed_files)}")
print(f"Skipped (>5GB): {len(skipped_large)}")
