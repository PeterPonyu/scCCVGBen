#!/usr/bin/env python3
"""run_d1_multiseed.py — D1 multi-seed stability driver (R1.3 reviewer response).

Demonstrates that centroid inference reduces variability across random seeds by
running the full 14-encoder sweep on a fixed set of datasets under multiple
random seeds.

Each (dataset, seed) combination is dispatched as a call to run_encoder_sweep.py
with SCCCVGBEN_SEED=<seed> set in the subprocess environment.  Output CSVs land
in results/multiseed/<dataset>__seed<N>.csv so they never overwrite the
canonical encoder_sweep results.

Usage:
  python scripts/run_d1_multiseed.py                        # dry-run, print commands
  python scripts/run_d1_multiseed.py --execute              # actually run
  python scripts/run_d1_multiseed.py --seeds 0,42 --workers 2 --execute
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ENCODERS_ALL = [
    "GAT", "GATv2", "Transformer", "SuperGAT",
    "GCN", "SAGE", "Graph", "Cheb",
    "TAG", "ARMA", "SG", "SSG",
    "GIN", "EdgeConv",
]

DEFAULT_DATASETS = [
    "GSE183904_GastricHmCancer",
    "GSE226131_HSCMmAged",
    "GSE128033_new",
]
DEFAULT_SEEDS = [0, 42, 123, 2024, 2026]
DEFAULT_WORKERS = 4
DEFAULT_EPOCHS = 100

MEAN_JOB_SECONDS = 60  # measured from prior sweep


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_h5ad(dataset_stem: str) -> Path:
    """Return path to h5ad or raise FileNotFoundError."""
    p = REPO_ROOT / "workspace" / "data" / "scrna" / f"{dataset_stem}.h5ad"
    if not p.exists():
        raise FileNotFoundError(f"h5ad not found: {p}")
    return p


def _build_command(
    h5ad_path: Path,
    seed: int,
    encoders: list[str],
    epochs: int,
    out_csv: Path,
) -> list[str]:
    """Build the subprocess argv for one (dataset, seed) job."""
    dataset_key = h5ad_path.stem
    # We pass --datasets-glob as an exact path (no wildcards) by using the
    # parent dir with a stem-exact pattern.  The glob in run_encoder_sweep
    # resolves relative to REPO_ROOT so we use a path relative to REPO_ROOT.
    rel_h5ad = h5ad_path.relative_to(REPO_ROOT)
    # run_encoder_sweep expects --out as a directory; seed-scoped dir per job
    out_dir = out_csv.parent
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_encoder_sweep.py"),
        "--datasets-glob", str(rel_h5ad),
        "--encoders", ",".join(encoders),
        "--epochs", str(epochs),
        "--out", str(out_dir),
    ]


def _env_for_seed(seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["SCCCVGBEN_SEED"] = str(seed)
    return env


def _print_estimate(n_jobs: int, workers: int) -> None:
    total_s = math.ceil(n_jobs * MEAN_JOB_SECONDS / workers)
    hours, rem = divmod(total_s, 3600)
    mins = rem // 60
    print(
        f"Estimated wall-clock: {n_jobs} jobs / {workers} workers "
        f"× {MEAN_JOB_SECONDS}s/job = ~{hours}h {mins:02d}m"
    )


# ---------------------------------------------------------------------------
# worker (called in subprocess pool)
# ---------------------------------------------------------------------------

def _run_job(argv: list[str], env: dict[str, str], dry_run: bool) -> int:
    """Execute one (dataset, seed) sweep; returns returncode."""
    if dry_run:
        return 0
    result = subprocess.run(argv, env=env)
    return result.returncode


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset stems (without .h5ad).  "
             f"Default: {','.join(DEFAULT_DATASETS)}",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated integer seeds.  "
             f"Default: {','.join(str(s) for s in DEFAULT_SEEDS)}",
    )
    parser.add_argument(
        "--encoders",
        default="all",
        help="'all' or comma-separated encoder names.  Default: all 14.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs.  Default: {DEFAULT_EPOCHS}.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel subprocess workers.  Default: {DEFAULT_WORKERS}.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch training.  Without this flag the script is a dry-run.",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    encoders = (
        ENCODERS_ALL if args.encoders.strip().lower() == "all"
        else [e.strip() for e in args.encoders.split(",") if e.strip()]
    )

    # Validate h5ad files exist
    h5ad_paths: list[Path] = []
    for ds in datasets:
        try:
            h5ad_paths.append(_resolve_h5ad(ds))
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}", file=sys.stderr)

    if not h5ad_paths:
        print("ERROR: no valid datasets found.", file=sys.stderr)
        sys.exit(1)

    out_base = REPO_ROOT / "results" / "multiseed"
    out_base.mkdir(parents=True, exist_ok=True)

    # Build job list: Cartesian product (dataset, seed)
    jobs: list[tuple[list[str], dict[str, str], Path]] = []
    for h5ad in h5ad_paths:
        for seed in seeds:
            out_csv = out_base / f"{h5ad.stem}__seed{seed}.csv"
            argv = _build_command(h5ad, seed, encoders, args.epochs, out_csv)
            env = _env_for_seed(seed)
            jobs.append((argv, env, out_csv))

    n_jobs = len(jobs)
    n_encoders = len(encoders)
    total_subjobs = n_jobs * n_encoders  # for estimate display

    print(
        f"D1 multi-seed stability sweep: "
        f"{len(h5ad_paths)} datasets × {len(seeds)} seeds × {n_encoders} encoders "
        f"= {total_subjobs} encoder-runs across {n_jobs} sweep invocations"
    )
    _print_estimate(total_subjobs, args.workers)
    print()

    if not args.execute:
        print("DRY-RUN — commands that would be launched (one per dataset×seed):")
        print()
        for argv, env, out_csv in jobs:
            seed_val = env["SCCCVGBEN_SEED"]
            print(f"  SCCCVGBEN_SEED={seed_val} \\")
            print(f"    {' '.join(argv)}")
            print(f"    # -> {out_csv.relative_to(REPO_ROOT)}")
            print()
        print("Re-run with --execute to launch.")
        return

    # Execute: partition jobs across workers
    print(f"Launching {n_jobs} sweep jobs across {args.workers} workers...")
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_job, argv, env, dry_run=False): (h5ad, seed)
            for (argv, env, out_csv), (h5ad, seed) in zip(
                jobs,
                [(h5ad, seed) for h5ad in h5ad_paths for seed in seeds],
            )
        }
        for fut in as_completed(futures):
            h5ad, seed = futures[fut]
            rc = fut.result()
            status = "OK" if rc == 0 else f"ERROR(rc={rc})"
            print(f"  [{status}] {h5ad.stem} seed={seed}")
            if rc != 0:
                errors.append(f"{h5ad.stem}/seed{seed}")

    if errors:
        print(f"\nFailed jobs ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"\nAll {n_jobs} jobs completed successfully.")
        print(f"Results in: {out_base.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
