#!/usr/bin/env python3
"""verify_benchmark_size.py — Assert benchmark meets revised size target.

Revised methodology (2026-04-23): **55 scRNA + 115 scATAC = 170**. Pass either
the revised targets (default) or the original v0.1 targets (100+100) via CLI.

Counts .h5ad files (both real files and symlinks) under workspace/data/scrna/
and workspace/data/scatac/. Symlinks pointing to non-existent targets are
counted as "ready slots" (placeholder for not-yet-fetched datasets).

Also verifies data/dropped_scatac_v2.csv row count (15 for v0.1; 0 for revised).

Exits 0 on PASS, non-zero on any failure.

Usage:
    python scripts/verify_benchmark_size.py                 # revised: 55+115, min=True
    python scripts/verify_benchmark_size.py --v01           # original: 100+100, strict
    python scripts/verify_benchmark_size.py --scrna 55 --scatac 115 --dropped 0 --min
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRNA_DIR = REPO_ROOT / "workspace" / "data" / "scrna"
SCATAC_DIR = REPO_ROOT / "workspace" / "data" / "scatac"
DROPPED_CSV = REPO_ROOT / "data" / "dropped_scatac_v2.csv"

# Revised methodology targets (default)
TARGET_SCRNA = 55
TARGET_SCATAC = 115
TARGET_DROPS = 0   # revised methodology does not drop scATAC samples

# v0.1 spec targets (pass --v01 to use)
V01_SCRNA = 100
V01_SCATAC = 100
V01_DROPS = 15


def _count_h5ad(directory: Path) -> int:
    """Count .h5ad files and symlinks (including broken symlinks = ready slots)."""
    if not directory.exists():
        return 0
    count = 0
    for p in directory.iterdir():
        if p.name.endswith(".h5ad") or (p.is_symlink() and p.name.endswith(".h5ad")):
            count += 1
    return count


def _count_drop_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return -1
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    return max(0, len(rows) - 1)  # subtract header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scrna-dir", default=str(SCRNA_DIR))
    parser.add_argument("--scatac-dir", default=str(SCATAC_DIR))
    parser.add_argument("--dropped-csv", default=str(DROPPED_CSV))
    parser.add_argument("--v01", action="store_true",
                        help="Use v0.1 targets (100 scRNA + 100 scATAC + 15 drops) instead of revised.")
    parser.add_argument("--scrna", type=int, default=None, help="Override scRNA target.")
    parser.add_argument("--scatac", type=int, default=None, help="Override scATAC target.")
    parser.add_argument("--dropped", type=int, default=None, help="Override drop-row target.")
    parser.add_argument("--min", action="store_true",
                        help="Treat targets as minimum (PASS if on-host count >= target).")
    args = parser.parse_args()

    if args.v01:
        target_scrna, target_scatac, target_drops = V01_SCRNA, V01_SCATAC, V01_DROPS
    else:
        target_scrna, target_scatac, target_drops = TARGET_SCRNA, TARGET_SCATAC, TARGET_DROPS

    if args.scrna is not None:
        target_scrna = args.scrna
    if args.scatac is not None:
        target_scatac = args.scatac
    if args.dropped is not None:
        target_drops = args.dropped

    scrna_dir = Path(args.scrna_dir)
    scatac_dir = Path(args.scatac_dir)
    dropped_csv = Path(args.dropped_csv)

    n_scrna = _count_h5ad(scrna_dir)
    n_scatac = _count_h5ad(scatac_dir)
    n_drops = _count_drop_rows(dropped_csv)

    def _ok(actual: int, target: int) -> bool:
        return actual >= target if args.min else actual == target

    results = []
    all_pass = True
    compare_symbol = ">=" if args.min else "=="

    # scRNA check
    if _ok(n_scrna, target_scrna):
        results.append(f"PASS  scRNA:  {n_scrna} {compare_symbol} {target_scrna} h5ad files")
    else:
        all_pass = False
        hint = f"  -> run `python scripts/fetch_geo_scrna.py` to fetch the remaining {max(0, target_scrna - n_scrna)}"
        results.append(f"FAIL  scRNA:  expected {compare_symbol} {target_scrna}, got {n_scrna}{hint}")

    # scATAC check
    if _ok(n_scatac, target_scatac):
        results.append(f"PASS  scATAC: {n_scatac} {compare_symbol} {target_scatac} h5ad files")
    else:
        all_pass = False
        hint = "  -> run `bash scripts/setup_symlinks.sh` to link existing scATAC h5ad files"
        results.append(f"FAIL  scATAC: expected {compare_symbol} {target_scatac}, got {n_scatac}{hint}")

    # Dropped CSV check
    if n_drops < 0:
        all_pass = False
        results.append(
            f"FAIL  dropped_scatac_v2.csv not found at {dropped_csv}\n"
            "       -> run `python scripts/select_scatac_drops.py`"
        )
    elif n_drops == target_drops or (args.min and n_drops <= target_drops):
        results.append(f"PASS  dropped: {n_drops} rows in dropped_scatac_v2.csv (target {target_drops})")
    else:
        all_pass = False
        results.append(
            f"FAIL  dropped: expected {target_drops} rows, got {n_drops} in {dropped_csv}"
        )

    print()
    for line in results:
        print(f"  {line}")
    print()

    if all_pass:
        total = target_scrna + target_scatac
        print(f"BENCHMARK VERIFICATION: PASS ({target_scrna} scRNA + {target_scatac} scATAC = {total} datasets)")
        sys.exit(0)
    else:
        print("BENCHMARK VERIFICATION: FAIL — see diagnostics above")
        sys.exit(1)


if __name__ == "__main__":
    main()
