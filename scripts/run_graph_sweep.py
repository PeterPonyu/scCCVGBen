#!/usr/bin/env python3
"""run_graph_sweep.py — Axis B: GAT × 5 graph constructions × all datasets.

Project-specific graph-axis sweep: explores how graph construction (backbone of
message-passing) affects latent quality when encoder is fixed to GAT. The
default kNN-euclidean graph is expanded to 5 graph-construction choices.

Graph methods (all use scccvgben.graphs.construction, k=15):
  1. kNN_euclidean       (shared with Axis A GAT cell — SKIPPED by default)
  2. kNN_cosine          (cosine-similarity kNN)
  3. snn                 (shared-nearest-neighbour)
  4. mutual_knn          (mutual kNN, stricter)
  5. gaussian_threshold  (Gaussian kernel with threshold)

Outputs:
  One CSV per dataset at results/graph_sweep/{dataset_key}.csv, 27-col schema,
  rows = 4 graph methods with method = scCCVGBen_GAT_{graph}.

Usage:
  python scripts/run_graph_sweep.py                      # 4 graphs × N datasets
  python scripts/run_graph_sweep.py --smoke              # 2 datasets × 2 graphs
  python scripts/run_graph_sweep.py --graphs kNN_cosine  # one graph only
"""
from __future__ import annotations

import argparse
import csv
import glob
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

GRAPH_METHODS = ["kNN_euclidean", "kNN_cosine", "snn", "mutual_knn", "gaussian_threshold"]
# kNN_euclidean is the Axis-A shared cell — skip here to avoid duplicating.
SKIP_GRAPHS = {"kNN_euclidean"}

from scccvgben.training.metrics import METRIC_COLS


def _run_one(h5ad_path: Path, graph_method: str, epochs: int, k: int) -> dict:
    from scccvgben.training.graph_sweep import run_scccvgben_graph_one
    return run_scccvgben_graph_one(
        h5ad_path=h5ad_path,
        graph_method=graph_method,
        method_name=f"scCCVGBen_GAT_{graph_method}",
        data_type="trajectory",
        epochs=epochs,
        k=k,
        silent=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-glob", default="workspace/data/scrna/*.h5ad")
    parser.add_argument("--graphs", default="all",
                        help="'all' (4 non-shared graphs) or comma-separated names.")
    parser.add_argument("--out", default="results/graph_sweep/")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (reference benchmark default=100).")
    parser.add_argument("--k", type=int, default=15, help="k for kNN-based graphs.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shard", default="0/1",
                        help="Shard 'i/N' for N-way parallelism (hash-based).")
    args = parser.parse_args()
    shard_i, shard_n = [int(x) for x in args.shard.split("/")]

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.graphs == "all":
        graphs = [g for g in GRAPH_METHODS if g not in SKIP_GRAPHS]
    else:
        graphs = [g.strip() for g in args.graphs.split(",")]

    if args.smoke:
        graphs = graphs[:2]
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

    log.info("Axis B: %d datasets × %d graphs (epochs=%d, k=%d)",
             len(dataset_paths), len(graphs), epochs, args.k)

    n_done = n_skip_graph = n_err = 0
    for ds_path in dataset_paths:
        ds_path = Path(ds_path)
        dataset_key = ds_path.stem
        out_csv = out_dir / f"{dataset_key}.csv"

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

        for graph_method in graphs:
            method = f"scCCVGBen_GAT_{graph_method}"
            if method in done_methods:
                n_skip_graph += 1
                continue

            t0 = time.time()
            try:
                row = _run_one(ds_path, graph_method, epochs, args.k)
            except Exception as exc:
                log.warning("  ✗ %s / %s failed: %s", dataset_key, graph_method, exc)
                n_err += 1
                continue

            is_new = not out_csv.exists() or out_csv.stat().st_size == 0
            with open(out_csv, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=METRIC_COLS, extrasaction="ignore")
                if is_new:
                    w.writeheader()
                w.writerow(row)
            done_methods.add(method)
            n_done += 1
            log.info("  ✓ %s / %s (%.0fs) ASW=%.3f NMI=%.3f [row appended]",
                     dataset_key, graph_method, time.time() - t0,
                     row.get("ASW", float("nan")),
                     row.get("NMI", float("nan")))

    log.info("Axis B complete: %d runs, %d per-graph skips (resumed), %d errors",
             n_done, n_skip_graph, n_err)


if __name__ == "__main__":
    main()
