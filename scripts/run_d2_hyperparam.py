#!/usr/bin/env python3
"""run_d2_hyperparam.py — D2: One-At-a-Time (OAT) hyperparameter sensitivity sweep.

Reviewer R2.2 asks for sensitivity analysis on beta, alpha, w_adj, dropout,
hidden_dim.  Each axis is swept while all other hyperparameters are held at
their default (Methods section) values.

OAT sweep grid (13 settings total including baseline):
  beta        : {0.5, [1.0], 2.0, 4.0}       -- KL weight
  alpha       : {0.1, [0.5], 1.0}             -- centroid coupling
  w_adj       : {0.0, 0.5, [1.0], 2.0}        -- adjacency reconstruction weight
  dropout     : {0.0, [0.05], 0.2}            -- encoder dropout
  hidden_dim  : {64, [128], 256}              -- hidden layer width
  (defaults in brackets)
  Non-default settings: 3+2+3+2+2 = 12 + 1 baseline = 13 settings
  × 14 encoders × 1 dataset = 182 jobs

Environment variables used by scccvgben_runner.py:
  SCCCVGBEN_BETA, SCCCVGBEN_ALPHA, SCCCVGBEN_W_ADJ,
  SCCCVGBEN_DROPOUT, SCCCVGBEN_HIDDEN_DIM

Output: results/hyperparam/<axis>__<value>__<dataset>.csv

Usage:
    python scripts/run_d2_hyperparam.py [--dataset GSE183904_GastricHmCancer]
                                        [--workers 4] [--execute]

Estimated wall-clock:
    60 s/job × 182 jobs / 4 workers ≈ 45 min
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Encoder set — 14 encoders matching Axis A
# ---------------------------------------------------------------------------
ENCODERS = [
    "GAT", "GATv2", "Transformer", "SuperGAT",
    "GCN", "SAGE", "Graph", "Cheb",
    "TAG", "ARMA", "SG", "SSG",
    "GIN", "EdgeConv",
]

# ---------------------------------------------------------------------------
# Hyperparameter defaults (Methods section / Table 1)
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "beta":       1.0,
    "alpha":      0.5,
    "w_adj":      1.0,
    "dropout":    0.05,
    "hidden_dim": 128,
}

# OAT sweep grid — each axis lists ALL values including the default
SWEEP_GRID: dict[str, list[Any]] = {
    "beta":       [0.5, 1.0, 2.0, 4.0],
    "alpha":      [0.1, 0.5, 1.0],
    "w_adj":      [0.0, 0.5, 1.0, 2.0],
    "dropout":    [0.0, 0.05, 0.2],
    "hidden_dim": [64, 128, 256],
}

# Map from axis name to the environment variable read by scccvgben_runner.py
ENV_VAR_MAP: dict[str, str] = {
    "beta":       "SCCCVGBEN_BETA",
    "alpha":      "SCCCVGBEN_ALPHA",
    "w_adj":      "SCCCVGBEN_W_ADJ",
    "dropout":    "SCCCVGBEN_DROPOUT",
    "hidden_dim": "SCCCVGBEN_HIDDEN_DIM",
}

# Metric columns produced by scccvgben_runner (27-col schema)
METRIC_COLUMNS = [
    "method", "axis", "param_name", "param_value",
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


# ---------------------------------------------------------------------------
# Job construction
# ---------------------------------------------------------------------------

def _build_jobs(dataset: str) -> list[dict]:
    """Return the full list of (axis, value, encoder) job dicts.

    The baseline run (all defaults) is represented as axis='baseline',
    value=None.  All 12 non-default settings are represented as separate
    jobs with their own axis/value pair.
    """
    jobs: list[dict] = []

    # Baseline: all defaults, every encoder
    for enc in ENCODERS:
        jobs.append({
            "axis": "baseline",
            "param_name": "baseline",
            "param_value": None,
            "encoder": enc,
            "env_overrides": {},
            "dataset": dataset,
        })

    # OAT non-default settings
    for axis, values in SWEEP_GRID.items():
        default_val = DEFAULTS[axis]
        for val in values:
            if val == default_val:
                continue  # default already covered by baseline
            env_key = ENV_VAR_MAP[axis]
            # Build env_overrides: all axes at default + this axis at val
            env_overrides = {ENV_VAR_MAP[k]: str(DEFAULTS[k]) for k in DEFAULTS}
            env_overrides[env_key] = str(val)
            for enc in ENCODERS:
                jobs.append({
                    "axis": axis,
                    "param_name": axis,
                    "param_value": val,
                    "encoder": enc,
                    "env_overrides": env_overrides,
                    "dataset": dataset,
                })

    return jobs


def _partition_jobs(jobs: list[dict], n_workers: int) -> list[list[dict]]:
    """Partition jobs into n_workers shards (round-robin)."""
    shards: list[list[dict]] = [[] for _ in range(n_workers)]
    for i, job in enumerate(jobs):
        shards[i % n_workers].append(job)
    return shards


# ---------------------------------------------------------------------------
# Worker — runs in a subprocess
# ---------------------------------------------------------------------------

_WORKER_ENTRYPOINT = """
import os, sys, csv, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hyperparam_worker")

REPO_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

import json, pickle

jobs_path = sys.argv[1]
out_dir   = Path(sys.argv[2])

with open(jobs_path, "rb") as fh:
    jobs = pickle.load(fh)

for job in jobs:
    axis       = job["axis"]
    param_name = job["param_name"]
    param_val  = job["param_value"]
    encoder    = job["encoder"]
    dataset    = job["dataset"]
    env_ovr    = job["env_overrides"]
    h5ad_path  = job["h5ad_path"]

    # Apply env overrides for this job
    saved_env = {}
    for k, v in env_ovr.items():
        saved_env[k] = os.environ.get(k)
        os.environ[k] = v

    # Also clear any stale env vars when running the baseline
    if axis == "baseline":
        for ev in ["SCCCVGBEN_BETA", "SCCCVGBEN_ALPHA", "SCCCVGBEN_W_ADJ",
                   "SCCCVGBEN_DROPOUT", "SCCCVGBEN_HIDDEN_DIM"]:
            saved_env.setdefault(ev, os.environ.get(ev))
            os.environ.pop(ev, None)

    label = f"{axis}__{param_val}__{encoder}__{dataset}"
    log.info("RUN %s", label)
    try:
        from scccvgben.training.scccvgben_runner import run_scccvgben_one
        metrics = run_scccvgben_one(
            h5ad_path=h5ad_path,
            graph_type=encoder,
            method_name=f"scCCVGBen_{encoder}",
        )
        metrics["axis"]        = axis
        metrics["param_name"]  = param_name
        metrics["param_value"] = str(param_val) if param_val is not None else "default"
    except Exception as exc:
        log.error("FAILED %s: %s", label, exc)
        metrics = {
            "method":      f"scCCVGBen_{encoder}",
            "axis":        axis,
            "param_name":  param_name,
            "param_value": str(param_val) if param_val is not None else "default",
        }

    # Restore env
    for k, orig in saved_env.items():
        if orig is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = orig

    # Write output CSV (one file per axis__value__dataset)
    val_str = str(param_val) if param_val is not None else "default"
    out_csv = out_dir / f"{axis}__{val_str}__{dataset}.csv"
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
    log.info("Wrote -> %s", out_csv)

log.info("Worker done: %d jobs", len(jobs))
"""


def _run_shard_subprocess(shard_jobs: list[dict], shard_idx: int,
                           out_dir: Path, dry_run: bool) -> subprocess.Popen | None:
    """Serialize shard jobs and launch a worker subprocess."""
    import pickle, tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="hyperparam_shard_"))
    jobs_path = tmp_dir / "jobs.pkl"
    worker_script = tmp_dir / "worker.py"

    with open(jobs_path, "wb") as fh:
        pickle.dump(shard_jobs, fh)

    with open(worker_script, "w") as fh:
        fh.write(_WORKER_ENTRYPOINT)

    cmd = [sys.executable, str(worker_script), str(jobs_path), str(out_dir)]
    log.info("SHARD %d: %d jobs -> %s", shard_idx, len(shard_jobs), " ".join(cmd))

    if dry_run:
        return None

    env = os.environ.copy()
    return subprocess.Popen(cmd, env=env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="GSE183904_GastricHmCancer",
        help="Dataset stem (h5ad filename without .h5ad). "
             "Default: GSE183904_GastricHmCancer",
    )
    parser.add_argument(
        "--h5ad-dir", default="workspace/data/scrna",
        help="Directory containing h5ad files (relative to repo root).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel subprocess workers (default: 4).",
    )
    parser.add_argument(
        "--out", default="results/hyperparam",
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually launch training jobs. Without this flag, runs a dry-run "
             "that prints the sweep grid and exits.",
    )
    args = parser.parse_args()

    dataset = args.dataset
    h5ad_dir = REPO_ROOT / args.h5ad_dir
    h5ad_path = h5ad_dir / f"{dataset}.h5ad"

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    dry_run = not args.execute

    # Build full job list
    jobs = _build_jobs(dataset)

    # Attach h5ad path to each job
    for j in jobs:
        j["h5ad_path"] = str(h5ad_path)

    # Summary
    n_settings = 1 + sum(
        len([v for v in vals if v != DEFAULTS[ax]])
        for ax, vals in SWEEP_GRID.items()
    )
    n_jobs = len(jobs)
    n_workers = args.workers
    est_min = math.ceil(60 * n_jobs / n_workers / 60)

    print(f"\n{'='*60}")
    print(f"  D2 OAT Hyperparameter Sweep")
    print(f"{'='*60}")
    print(f"  Dataset  : {dataset}")
    print(f"  Settings : {n_settings}  (1 baseline + 12 non-default)")
    print(f"  Encoders : {len(ENCODERS)}")
    print(f"  Total jobs: {n_jobs}  ({n_settings} settings × {len(ENCODERS)} encoders)")
    print(f"  Workers  : {n_workers}")
    print(f"  Est. wall-clock: {est_min} min  (at 60 s/job)")
    print(f"  Output   : {out_dir}/")
    print(f"  Mode     : {'EXECUTE' if args.execute else 'DRY-RUN (pass --execute to run)'}")
    print(f"{'='*60}\n")

    # Print sweep grid
    print("Sweep grid:")
    print(f"  {'axis':<12} {'values':<30} {'default'}")
    print(f"  {'-'*12} {'-'*30} {'-'*10}")
    for axis, vals in SWEEP_GRID.items():
        vals_str = "{" + ", ".join(
            f"[{v}]" if v == DEFAULTS[axis] else str(v)
            for v in vals
        ) + "}"
        print(f"  {axis:<12} {vals_str:<30} {DEFAULTS[axis]}")

    # Partition and display shards
    shards = _partition_jobs(jobs, n_workers)
    print(f"\n4-shard partitioning ({n_jobs} jobs across {n_workers} workers):")
    for i, shard in enumerate(shards):
        print(f"  Shard {i}: {len(shard)} jobs")

    # Show first 10 jobs
    print(f"\nFirst 10 jobs (of {n_jobs}):")
    for j in jobs[:10]:
        val_str = str(j["param_value"]) if j["param_value"] is not None else "default"
        print(f"  [{j['axis']:<12} val={val_str:<6}] encoder={j['encoder']}")

    if dry_run:
        print(f"\nDRY-RUN complete. Pass --execute to launch {n_jobs} training jobs.")
        return

    # Execute: launch one subprocess per shard
    log.info("Launching %d shards with %d total jobs ...", n_workers, n_jobs)
    t0 = time.time()

    procs: list[subprocess.Popen] = []
    for i, shard in enumerate(shards):
        if not shard:
            continue
        proc = _run_shard_subprocess(shard, i, out_dir, dry_run=False)
        if proc is not None:
            procs.append(proc)

    # Wait for all workers
    for proc in procs:
        proc.wait()

    elapsed = time.time() - t0
    log.info("All shards complete in %.1f s (%.1f min)", elapsed, elapsed / 60)

    # Collect output CSVs
    csvs = sorted(out_dir.glob("*.csv"))
    log.info("Output files (%d):", len(csvs))
    for p in csvs:
        log.info("  %s", p)


if __name__ == "__main__":
    main()
