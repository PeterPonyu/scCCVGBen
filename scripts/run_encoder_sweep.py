#!/usr/bin/env python3
"""run_encoder_sweep.py — Axis A: 14 encoders × all datasets.

Reads workspace/data/scrna/*.h5ad (unified view: on-host + GEO).
Writes one CSV per dataset to results/encoder_sweep/{dataset_key}.csv.

Each row is a (method=scCCVGBen_{encoder}, ...) record in the scCCVGBen
CG_dl_merged 27-column schema, computed via the vendored reference core
(scccvgben.external.reference_core.cgvae.CGVAE_agent) — bit-compatible with
reused results.

Reuse:
  GAT × reused scRNA cells are symlinked from CG_dl_merged via
  workspace/reused_results/axisA_GAT_scrna/ and skipped here to avoid
  re-running. See results/encoder_sweep/README.md for GAT-row provenance.

Usage:
  python scripts/run_encoder_sweep.py                     # full sweep
  python scripts/run_encoder_sweep.py --smoke             # 2 datasets × 2 encoders
  python scripts/run_encoder_sweep.py --encoders GAT,GCN  # subset
"""
from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
REUSED_GAT_DIR = REPO_ROOT / "workspace" / "reused_results" / "axisA_GAT_scrna"

# Axis A encoder set — 10 reference encoder families plus 4 project-specific
# variants (GATv2, SuperGAT, GIN, EdgeConv) for broader architecture comparison.
# Names match the extended CGVAE_module.CONV_LAYERS keys.
ENCODERS = [
    # Attention family (4)
    "GAT",           # default attention encoder
    "GATv2",         # project extension — dynamic attention (Brody 2022)
    "Transformer",   # attention transformer convolution
    "SuperGAT",      # project extension — self-supervised edge prediction
    # Message-passing family (8)
    "GCN", "SAGE", "Graph", "Cheb",
    "TAG", "ARMA", "SG", "SSG",          # reference encoder set
    "GIN", "EdgeConv",                   # project extensions
]

from scccvgben.training.metrics import METRIC_COLS


def _old_scrna_keys() -> set[str]:
    """Dataset stems already covered in CG_dl_merged (GAT row reused via symlink)."""
    if not REUSED_GAT_DIR.exists():
        return set()
    return {p.stem.replace("_df", "") for p in REUSED_GAT_DIR.glob("*_df.csv")}


def _run_one(h5ad_path: Path, encoder_name: str, epochs: int) -> dict:
    """Run scCCVGBen on one dataset with one encoder. Returns 27-col dict."""
    from scccvgben.training.scccvgben_runner import run_scccvgben_one
    return run_scccvgben_one(
        h5ad_path=h5ad_path,
        graph_type=encoder_name,
        method_name=f"scCCVGBen_{encoder_name}",
        data_type="trajectory",
        epochs=epochs,
        silent=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-glob", default="workspace/data/scrna/*.h5ad")
    parser.add_argument("--encoders", default="all",
                        help="'all' or comma-separated encoder names.")
    parser.add_argument("--out", default="results/encoder_sweep/")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (reference benchmark default=100).")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only first 2 datasets x first 2 encoders for 5 epochs.")
    parser.add_argument("--shard", default="0/1",
                        help="Shard this worker processes (format 'i/N': worker i of N). "
                             "Datasets split by hash(key) %% N == i. Enables N-way parallelism.")
    args = parser.parse_args()
    shard_i, shard_n = [int(x) for x in args.shard.split("/")]

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    encoders = ENCODERS if args.encoders == "all" else [e.strip() for e in args.encoders.split(",")]
    if args.smoke:
        encoders = encoders[:2]
        epochs = 5
    else:
        epochs = args.epochs

    pattern = str(REPO_ROOT / args.datasets_glob)
    dataset_paths = sorted(glob.glob(pattern))
    if shard_n > 1:
        dataset_paths = [p for p in dataset_paths if (hash(Path(p).stem) % shard_n) == shard_i]
        log.info("shard %d/%d: %d datasets", shard_i, shard_n, len(dataset_paths))
    if args.smoke:
        dataset_paths = dataset_paths[:2]

    reused_gat_keys = _old_scrna_keys()
    log.info("Axis A: %d datasets × %d encoders (epochs=%d, reused GAT keys=%d)",
             len(dataset_paths), len(encoders), epochs, len(reused_gat_keys))

    n_done = n_skip_encoder = n_err = 0
    for ds_path in dataset_paths:
        ds_path = Path(ds_path)
        dataset_key = ds_path.stem
        out_csv = out_dir / f"{dataset_key}.csv"

        # Per-encoder resume: read any existing rows to determine already-done methods
        done_methods: set[str] = set()
        if out_csv.exists() and out_csv.stat().st_size > 0:
            try:
                with open(out_csv, newline="") as fh:
                    reader = csv.DictReader(fh)
                    for r in reader:
                        m = r.get("method", "").strip()
                        if m: done_methods.add(m)
            except Exception:
                pass

        for encoder in encoders:
            # Axis A reuse: skip GAT × known reused scRNA (reuse from CG_dl_merged)
            if encoder == "GAT" and dataset_key in reused_gat_keys:
                continue
            method = f"scCCVGBen_{encoder}"
            if method in done_methods:
                n_skip_encoder += 1
                continue

            t0 = time.time()
            try:
                row = _run_one(ds_path, encoder, epochs)
            except Exception as exc:
                log.warning("  ✗ %s / %s failed: %s", dataset_key, encoder, exc)
                n_err += 1
                continue

            # Atomic append: write header if new file, else append row.
            is_new = not out_csv.exists() or out_csv.stat().st_size == 0
            with open(out_csv, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=METRIC_COLS, extrasaction="ignore")
                if is_new:
                    w.writeheader()
                w.writerow(row)
            done_methods.add(method)
            n_done += 1
            log.info("  ✓ %s / %s (%.0fs) ASW=%.3f NMI=%.3f [row appended]",
                     dataset_key, encoder, time.time() - t0,
                     row.get("ASW", float("nan")),
                     row.get("NMI", float("nan")))

    log.info("Axis A complete: %d runs, %d per-encoder skips (resumed), %d errors",
             n_done, n_skip_encoder, n_err)


if __name__ == "__main__":
    main()
