#!/usr/bin/env python3
"""run_baseline_backfill.py — Axis C: baseline backfill for the 45 NEW scRNA datasets.

Only runs on NEW scRNA (not the 55 old scRNA whose baselines are already in
workspace/reused_results/scrna_baselines/). For each new dataset and each
baseline method, calls scccvgben.baselines.run_baseline() and writes a CSV.

Output: results/baselines/scrna_{dataset_key}.csv
  Columns: method, <27 metrics>

Usage:
    python scripts/run_baseline_backfill.py \\
        --scrna-glob 'workspace/data/scrna/*.h5ad' \\
        --baselines all \\
        --out results/baselines/ \\
        [--new-scrna-only] \\
        [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
REUSED_SCRNA_DIR = REPO_ROOT / "workspace" / "reused_results" / "scrna_baselines"

METRIC_COLUMNS = [
    "method", "ASW", "DAV", "CAL", "COR",
    "distance_correlation_umap", "Q_local_umap", "Q_global_umap",
    "K_max_umap", "overall_quality_umap",
    "distance_correlation_tsne", "Q_local_tsne", "Q_global_tsne",
    "K_max_tsne", "overall_quality_tsne",
    "manifold_dimensionality_intrin", "spectral_decay_rate_intrin",
    "participation_ratio_intrin", "anisotropy_score_intrin",
    "trajectory_directionality_intrin", "noise_resilience_intrin",
    "core_quality_intrin", "overall_quality_intrin",
    "data_type_intrin", "interpretation_intrin",
    "NMI", "ARI",
]

ALL_BASELINES = [
    "PCA", "KPCA", "ICA", "FA", "NMF", "TSVD", "DICL",
    "scVI", "DIP", "INFO", "TC", "highBeta", "CCVGAE",
]


def _old_scrna_keys() -> set[str]:
    """Return the stems of old scRNA datasets (have reused baseline CSVs)."""
    if not REUSED_SCRNA_DIR.exists():
        return set()
    return {p.stem for p in REUSED_SCRNA_DIR.glob("*.csv")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scrna-glob", default="workspace/data/scrna/*.h5ad",
                        help="Glob for scRNA h5ad files.")
    parser.add_argument("--baselines", default="all",
                        help="'all' or comma-separated baseline names.")
    parser.add_argument("--out", default="results/baselines/",
                        help="Output directory for per-dataset CSVs.")
    parser.add_argument("--new-scrna-only", action="store_true", default=True,
                        help="Skip datasets that already have reused baselines (default: True).")
    parser.add_argument("--smoke", action="store_true",
                        help="Run PCA on the first 1 new dataset only.")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from scccvgben.baselines import run_baseline

    if args.baselines == "all":
        baselines = ALL_BASELINES
    else:
        baselines = [b.strip() for b in args.baselines.split(",")]

    pattern = str(REPO_ROOT / args.scrna_glob)
    h5ad_files = sorted(glob.glob(pattern, recursive=True))

    if not h5ad_files:
        log.error("No h5ad files found: %s", pattern)
        sys.exit(1)

    old_keys = _old_scrna_keys() if args.new_scrna_only else set()

    if args.smoke:
        # Find first new dataset, run PCA only
        h5ad_files = [
            f for f in h5ad_files if Path(f).stem not in old_keys
        ][:1]
        baselines = ["PCA"]
        log.info("--smoke: 1 new dataset, PCA only.")

    if not h5ad_files:
        log.info("No new scRNA h5ad files to process (all covered by reused baselines).")
        return

    total = 0
    for h5ad_path in map(Path, h5ad_files):
        dataset_key = h5ad_path.stem

        if dataset_key in old_keys:
            log.info("SKIP (reused): %s", dataset_key)
            continue

        out_csv = out_dir / f"scrna_{dataset_key}.csv"
        rows: list[dict] = []

        for method_name in baselines:
            log.info("RUN baseline %s on %s", method_name, dataset_key)
            try:
                metrics = run_baseline(method_name, h5ad_path, modality="scrna")
                rows.append(metrics)
            except Exception as exc:
                log.error("FAILED %s / %s: %s", dataset_key, method_name, exc)
                rows.append({"method": method_name})

        if rows:
            with open(out_csv, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            log.info("Wrote %d rows -> %s", len(rows), out_csv)
            total += len(rows)

    log.info("Baseline backfill complete. Total rows: %d", total)


if __name__ == "__main__":
    main()
