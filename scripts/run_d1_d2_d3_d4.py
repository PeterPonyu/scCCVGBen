#!/usr/bin/env python3
"""run_d1_d2_d3_d4.py — Unified orchestrator for the 4 reviewer-driven experiments.

Runs in sequence (default) or in parallel where GPU contention permits:
  D1: multi-seed stability (3 datasets × 5 seeds × 14 encoders = 210 jobs, ~52 min)
  D2: hyperparameter OAT (1 dataset × 13 settings × 14 encoders = 182 jobs, ~46 min)
  D3: linear-encoder ablation — RESOLVED via existing PCA/KPCA/FA/NMF/TSVD/DICL
      baselines in fig07; no new GPU run needed (see rebuttal letter R2.4).
  D4: SOTA baselines (scGPT, scFoundation) — requires manual weight downloads;
      see scripts/run_d4_sota_baselines.py docstring for prereqs.

Default behaviour: DRY-RUN — prints the launch sequence and time estimates.
Pass --execute to actually launch.

D1 + D2 share the single GPU; running them concurrently would over-subscribe
VRAM. We run D1 first (~52 min) then D2 (~46 min) for ~1h 38m total wall
clock with --workers 4 each.

Usage:
    python scripts/run_d1_d2_d3_d4.py                    # dry-run plan
    python scripts/run_d1_d2_d3_d4.py --execute          # serial D1 then D2
    python scripts/run_d1_d2_d3_d4.py --execute --skip-d4  # skip SOTA preflight
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _phase(name: str, est_min: int, started: datetime) -> None:
    elapsed = (datetime.now() - started).total_seconds() / 60
    print(f"\n[{datetime.now().strftime('%H:%M:%S')} | elapsed {elapsed:5.1f}m] === {name} (est {est_min} min) ===")


def _run(cmd: list[str], execute: bool) -> int:
    print(f"  $ {' '.join(cmd)}")
    if not execute:
        return 0
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true",
                        help="Actually launch experiments (default: dry-run plan only).")
    parser.add_argument("--workers", type=int, default=4,
                        help="GPU shard count for D1/D2 (default 4).")
    parser.add_argument("--skip-d1", action="store_true")
    parser.add_argument("--skip-d2", action="store_true")
    parser.add_argument("--skip-d4", action="store_true")
    args = parser.parse_args()

    started = datetime.now()
    print(f"Start: {started.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workers per phase: {args.workers}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")

    total_est = 0
    if not args.skip_d1:
        total_est += 52
    if not args.skip_d2:
        total_est += 46
    if not args.skip_d4:
        total_est += 5  # preflight only; actual D4 needs weights and is gated separately
    eta = started + timedelta(minutes=total_est)
    print(f"Estimated wall-clock: {total_est} min  (ETA {eta.strftime('%H:%M:%S')})")

    failures: list[str] = []

    if not args.skip_d1:
        _phase("D1 multi-seed stability", 52, started)
        rc = _run([sys.executable, str(SCRIPTS / "run_d1_multiseed.py"),
                   "--workers", str(args.workers), "--execute"], args.execute)
        if rc != 0:
            failures.append(f"D1 exited {rc}")

    if not args.skip_d2:
        _phase("D2 hyperparameter OAT", 46, started)
        rc = _run([sys.executable, str(SCRIPTS / "run_d2_hyperparam.py"),
                   "--workers", str(args.workers), "--execute"], args.execute)
        if rc != 0:
            failures.append(f"D2 exited {rc}")

    _phase("D3 linear-encoder", 0, started)
    print("  RESOLVED — fig07's PCA/KPCA/FA/NMF/TSVD/DICL baselines already cover")
    print("  the 'linear encoder alternative' the reviewer asked about (R2.4).")
    print("  No new GPU run; the rebuttal cites existing fig07 numbers.")

    if not args.skip_d4:
        _phase("D4 SOTA baselines preflight", 5, started)
        rc = _run([sys.executable, str(SCRIPTS / "run_d4_sota_baselines.py"),
                   "--workers", str(args.workers)], args.execute)
        if rc != 0:
            print(f"  [info] D4 preflight reported missing prereqs (rc={rc}) — see top-of-file docstring of run_d4_sota_baselines.py for install/weight instructions.")

    elapsed = (datetime.now() - started).total_seconds() / 60
    print(f"\n=== DONE | wall-clock {elapsed:.1f} min ===")
    if failures:
        for f in failures:
            print(f"  FAILED: {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
