#!/usr/bin/env python3
"""run_encoder_sweep_parallel.py — Parallel launcher for run_encoder_sweep.py.

Spawns N shards of run_encoder_sweep.py, each processing a disjoint subset of
datasets (split by hash(stem) % N == i), using the existing --shard i/N flag.

Usage:
    python scripts/run_encoder_sweep_parallel.py --workers 2
    python scripts/run_encoder_sweep_parallel.py --workers 4 --mem-fraction 0.22
    python scripts/run_encoder_sweep_parallel.py --workers 2 --epochs 50 \\
        --datasets-glob 'workspace/data/scrna/*.h5ad' --encoders GAT,GCN

Each shard's stdout+stderr is tee'd to:
    workspace/logs/sweep-<YYYYMMDD-HHMMSS>-shard<i>.log

The wrapper exits 0 only when all shards exit 0; otherwise exits 1 and
reports which shards failed.

Do NOT import this module — it is a script entry point only.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "run_encoder_sweep.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel shards (default: 4 — measured GPU util ~25%% at 2 shards on RTX 4080 Laptop).",
    )
    parser.add_argument(
        "--mem-fraction", type=float, default=0.22,
        help="CUDA memory fraction per shard (default: 0.22 — 4 shards × 0.22 = 0.88 of 12 GB VRAM).",
    )
    parser.add_argument(
        "--datasets-glob", default="workspace/data/scrna/*.h5ad",
        help="Glob pattern forwarded to each shard (default: workspace/data/scrna/*.h5ad). "
             "Multi-glob (e.g. {a,b}.h5ad) is NOT supported — launch the wrapper twice for two specific datasets.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs forwarded to each shard (default: 100).",
    )
    parser.add_argument(
        "--encoders", default="all",
        help="Encoder subset forwarded to each shard (default: all).",
    )
    args = parser.parse_args()

    n = args.workers
    log_dir = REPO_ROOT / "workspace" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    procs: list[tuple[int, subprocess.Popen, Path]] = []

    for i in range(n):
        log_path = log_dir / f"sweep-{stamp}-shard{i}.log"
        cmd = [
            sys.executable, str(SWEEP_SCRIPT),
            "--shard", f"{i}/{n}",
            "--datasets-glob", args.datasets_glob,
            "--epochs", str(args.epochs),
            "--encoders", args.encoders,
        ]
        shard_env = os.environ.copy()
        shard_env["CUDA_MEM_FRACTION"] = str(args.mem_fraction)

        log_fh = open(log_path, "w")  # noqa: WPS515 — kept open until proc finishes
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=shard_env,
        )
        procs.append((i, proc, log_path))
        print(f"[launcher] shard {i}/{n} pid={proc.pid}  log={log_path}")

    failed: list[int] = []
    for shard_i, proc, log_path in procs:
        rc = proc.wait()
        # Close the log file handle stored inside Popen's stdout
        if proc.stdout and not proc.stdout.closed:
            proc.stdout.close()
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"[launcher] shard {shard_i}/{n} {status}  log={log_path}")
        if rc != 0:
            failed.append(shard_i)

    if failed:
        print(f"[launcher] {len(failed)} shard(s) failed: {failed}", file=sys.stderr)
        sys.exit(1)
    print(f"[launcher] all {n} shards completed successfully.")


if __name__ == "__main__":
    main()
