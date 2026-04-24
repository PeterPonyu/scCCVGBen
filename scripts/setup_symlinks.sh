#!/usr/bin/env bash
# setup_symlinks.sh — Idempotent symlink setup for scCCVGBen workspace.
#
# Establishes symlinks from pre-computed CCVGAE result CSVs into:
#   workspace/reused_results/scrna_baselines/
#   workspace/reused_results/axisA_GAT_scrna/
#   workspace/reused_results/scatac_baselines/
#
# h5ad symlinks for scRNA + scATAC are created from SCRNA_SOURCES / SCATAC_SOURCES
# arrays (see below). Paths parameterised via REPO / SRC environment variables
# so the script is portable across hosts.
#
# Usage:
#   bash scripts/setup_symlinks.sh                       # defaults to /home/zeyufu/... paths
#   REPO=/other/path SRC=/ccvgae bash scripts/setup_symlinks.sh    # custom locations
#
# Safe to re-run: every link uses ln -sfn (force, no-dereference).
set -euo pipefail

# ── parameterised paths (env-overridable for portability) ────────────────────
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="${SRC:-/home/zeyufu/LAB/CCVGAE}"

# ── safety check ─────────────────────────────────────────────────────────────
if [[ ! -d "$REPO" ]]; then
    echo "ERROR: REPO=$REPO does not exist. Create the repo first or pass REPO=<path>." >&2
    exit 1
fi

WS="$REPO/workspace"

# ── create workspace directories ─────────────────────────────────────────────
mkdir -p "$WS/data/scrna"
mkdir -p "$WS/data/scatac"
mkdir -p "$WS/reused_results/scrna_baselines"
mkdir -p "$WS/reused_results/axisA_GAT_scrna"
mkdir -p "$WS/reused_results/scatac_baselines"
mkdir -p "$WS/geo_cache"
mkdir -p "$WS/checkpoints"
mkdir -p "$WS/logs"

# ── Reuse Map row 1: scRNA baseline CSVs (CG_dl_merged) ──────────────────────
SCRNA_SRC="$SRC/CG_results/CG_dl_merged"
scrna_count=0

if [[ ! -d "$SCRNA_SRC" ]]; then
    echo "WARNING: scRNA baseline source not found: $SCRNA_SRC" >&2
else
    for f in "$SCRNA_SRC"/*.csv; do
        [[ -f "$f" ]] || continue
        fname="$(basename "$f")"
        ln -sfn "$f" "$WS/reused_results/scrna_baselines/$fname"
        scrna_count=$((scrna_count + 1))
    done
fi

# ── Reuse Map row 2: Axis-A GAT scRNA rows ───────────────────────────────────
gat_count=0
if [[ -d "$SCRNA_SRC" ]]; then
    for f in "$SCRNA_SRC"/*.csv; do
        [[ -f "$f" ]] || continue
        fname="$(basename "$f")"
        ln -sfn "$f" "$WS/reused_results/axisA_GAT_scrna/$fname"
        gat_count=$((gat_count + 1))
    done
fi

# ── Reuse Map row 3: scATAC baseline CSVs (CG_atacs/tables) ──────────────────
SCATAC_SRC="$SRC/CG_results/CG_atacs/tables"
scatac_count=0

if [[ ! -d "$SCATAC_SRC" ]]; then
    echo "WARNING: scATAC baseline source not found: $SCATAC_SRC" >&2
else
    for f in "$SCATAC_SRC"/*.csv; do
        [[ -f "$f" ]] || continue
        fname="$(basename "$f")"
        ln -sfn "$f" "$WS/reused_results/scatac_baselines/$fname"
        scatac_count=$((scatac_count + 1))
    done
fi

# ── scRNA h5ad source directories ────────────────────────────────────────────
SCRNA_SOURCES=(
  "/home/zeyufu/LAB/DATA"
  "/home/zeyufu/Downloads/DevelopmentDatasets"
  "/home/zeyufu/Downloads/DevelopmentDatasets2"
  "/home/zeyufu/Downloads/CancerDatasets"
  "/home/zeyufu/Downloads/CancerDatasets2"
  "/home/zeyufu/LAB/SCRL"
  "/home/zeyufu/LAB/MCC_results"
  "/home/zeyufu/Desktop/HMJ"
  "/home/zeyufu/Desktop/CSD"
  "/home/zeyufu/Desktop/WC"
  "/home/zeyufu/Desktop/LSK-LCN-Publicdata"
  "/home/zeyufu/vGAE_LAB/data"
)

# ── scATAC h5ad source directories (resolved from T2 inventory) ──────────────
# NOTE: /home/zeyufu/Desktop/scATAC-25100/ contains 439 h5/h5ad files distributed
# across per-GSE subdirectories; we recurse one level in for those.
SCATAC_SOURCES=(
  "/home/zeyufu/Downloads/ATAC_data"
  "/home/zeyufu/Desktop/scATAC-25100"
)

# Load blocklist: filename stems excluded from benchmark (unverifiable metadata)
BLOCKLIST="${REPO:-$(dirname "$0")/..}/data/scrna_blocklist.txt"
declare -A blocked
if [ -f "$BLOCKLIST" ]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        blocked["$line"]=1
    done < "$BLOCKLIST"
fi

n_scrna_h5ad=0
n_scatac_h5ad=0
n_scrna_blocked=0

# scRNA h5ad (top-level in each source dir)
for src in "${SCRNA_SOURCES[@]}"; do
  if [ -d "$src" ]; then
    for f in "$src"/*.h5ad; do
      [ -f "$f" ] || continue
      bn=$(basename "$f")
      stem="${bn%.h5ad}"
      if [[ "$bn" == *atac* || "$bn" == *ATA* || "$bn" == *ATAC* ]]; then
        ln -sfn "$f" "$WS/data/scatac/$bn" && n_scatac_h5ad=$((n_scatac_h5ad+1)) || true
      elif [[ -n "${blocked[$stem]:-}" ]]; then
        n_scrna_blocked=$((n_scrna_blocked+1))
      else
        ln -sfn "$f" "$WS/data/scrna/$bn" && n_scrna_h5ad=$((n_scrna_h5ad+1)) || true
      fi
    done
  fi
done

# scATAC h5ad (recurse one level for scATAC-25100 per-GSE subdirs)
for src in "${SCATAC_SOURCES[@]}"; do
  if [ -d "$src" ]; then
    # top-level h5ad / h5
    for f in "$src"/*.h5ad "$src"/*.h5; do
      [ -f "$f" ] || continue
      bn=$(basename "$f")
      ln -sfn "$f" "$WS/data/scatac/$bn" && n_scatac_h5ad=$((n_scatac_h5ad+1)) || true
    done
    # one level down (scATAC-25100/{N}-{GSE}/*.h5)
    for sub in "$src"/*/; do
      [ -d "$sub" ] || continue
      for f in "$sub"/*.h5ad "$sub"/*.h5; do
        [ -f "$f" ] || continue
        bn="$(basename "$(dirname "$f")")__$(basename "$f")"
        ln -sfn "$f" "$WS/data/scatac/$bn" && n_scatac_h5ad=$((n_scatac_h5ad+1)) || true
      done
    done
  fi
done

if [[ $n_scrna_h5ad -eq 0 && $n_scatac_h5ad -eq 0 ]]; then
  echo "WARNING: No h5ad files found in any source directory. Check SCRNA_SOURCES / SCATAC_SOURCES paths." >&2
fi

# ── downloaded data: link scrna_geo/ + scatac_geo/ into scrna/ + scatac/ ─────
# fetch_geo_scrna.py writes to workspace/data/scrna_geo/ (raw downloads stay
# separated from on-host raw-data symlinks). This step ensures training scripts,
# which read from workspace/data/scrna/, see both sources through a unified view.
n_scrna_geo=0
n_scatac_geo=0
for f in "$WS"/data/scrna_geo/*.h5ad; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    ln -sfn "$f" "$WS/data/scrna/$bn" && n_scrna_geo=$((n_scrna_geo+1)) || true
done
for f in "$WS"/data/scatac_geo/*.h5ad "$WS"/data/scatac_geo/*.h5; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    ln -sfn "$f" "$WS/data/scatac/$bn" && n_scatac_geo=$((n_scatac_geo+1)) || true
done

# ── site image symlink directory (for Hugo static/) ──────────────────────────
mkdir -p "$REPO/site/static/images"
fig_count=0
for f in "$REPO"/figures/*.png; do
    [[ -e "$f" ]] || continue
    ln -sfn "../../figures/$(basename "$f")" "$REPO/site/static/images/$(basename "$f")"
    fig_count=$((fig_count + 1))
done

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== setup_symlinks.sh complete ==="
echo "  REPO: $REPO"
echo "  SRC : $SRC"
echo "  scRNA baseline CSVs linked : $scrna_count  (-> workspace/reused_results/scrna_baselines/)"
echo "  AxisA GAT scRNA CSVs linked: $gat_count   (-> workspace/reused_results/axisA_GAT_scrna/)"
echo "  scATAC baseline CSVs linked: $scatac_count (-> workspace/reused_results/scatac_baselines/)"
echo "  h5ad symlinks (on-host sources): scRNA=$n_scrna_h5ad, scATAC=$n_scatac_h5ad  (blocked scRNA=$n_scrna_blocked)"
echo "  h5ad symlinks (GEO downloads) : scRNA=$n_scrna_geo,  scATAC=$n_scatac_geo"
echo "  Figure symlinks for Hugo       : $fig_count"
echo ""
echo "Next steps:"
echo "  python scripts/build_datasets_csv.py --scan-host       # locate h5ad files"
echo "  python scripts/build_datasets_csv.py --build-canonical # build datasets.csv"
echo "  python scripts/verify_benchmark_size.py                # assert target benchmark size"
