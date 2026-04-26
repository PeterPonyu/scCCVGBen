#!/usr/bin/env python3
"""run_pair_sweep.py — Pair-wise variant ablation sweep.

For each (dataset, pair_name) combination, train two CentroidVAEAgents with
the pair-defined params, save:

  results/pair_sweep/{pair_name}/tables/{prefix}_{name}_df.csv
  results/pair_sweep/{pair_name}/series/{prefix}_{name}_dfs.csv

Layout matches the legacy-reference pair-output shape so downstream figure
code can consume the new outputs without adjustment.

Resume: skip a (dataset, pair) when both tables/ and series/ files exist.

Usage:
  python scripts/run_pair_sweep.py                                     # all 3 pairs × all .h5ad
  python scripts/run_pair_sweep.py --pairs VGAE_pair                   # one pair
  python scripts/run_pair_sweep.py --datasets-glob 'workspace/data/scrna/*.h5ad'
  python scripts/run_pair_sweep.py --epochs 50 --smoke                 # 2 ds × 1 pair × 50 epochs
"""
from __future__ import annotations

import argparse
import glob
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_prefix(stem: str) -> str:
    """Return a legacy-reference category prefix used in tables/{prefix}_{name}_df.csv.

    Mirrors the ``_infer_category`` heuristic in ``reconcile_result_schema``:
    ``Can`` (cancer), ``Dev`` (development), ``sup`` (supplement: irall/wtko),
    else ``Gen``.
    """
    lo = stem.lower()
    if stem in ("irall", "wtko", "hemato"):
        return "sup"
    if "cancer" in lo or "tumor" in lo or "mcc" in lo:
        return "Can"
    if "dev" in lo or "embryo" in lo or "esc" in lo or "hsc" in lo:
        return "Dev"
    return "Gen"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-glob", default="workspace/data/scrna/*.h5ad",
                        help="Glob for h5ad inputs.")
    parser.add_argument("--pairs", default="all",
                        help="'all' or comma-separated subset (VGAE_pair,CouVAE_pair,Linear_pair).")
    parser.add_argument("--out", default="results/pair_sweep/")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs per agent (legacy supplement default: 200).")
    parser.add_argument("--smoke", action="store_true",
                        help="2 datasets x 1 pair x 50 epochs.")
    args = parser.parse_args()

    from scccvgben.training.pair_sweep import PAIR_DEFINITIONS, run_pair_one

    if args.pairs == "all":
        pair_names = list(PAIR_DEFINITIONS)
    else:
        pair_names = [p.strip() for p in args.pairs.split(",")]
        unknown = [p for p in pair_names if p not in PAIR_DEFINITIONS]
        if unknown:
            raise SystemExit(f"Unknown pair(s): {unknown}. Choices: {list(PAIR_DEFINITIONS)}")

    pattern = str(REPO_ROOT / args.datasets_glob)
    dataset_paths = sorted(glob.glob(pattern))
    if args.smoke:
        dataset_paths = dataset_paths[:2]
        pair_names = pair_names[:1]
        epochs = 50
    else:
        epochs = args.epochs

    out_root = REPO_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("pair_sweep: %d datasets x %d pairs (epochs=%d) → %s",
             len(dataset_paths), len(pair_names), epochs, out_root)

    n_done = n_skip = n_err = 0
    for pair_name in pair_names:
        tables_dir = out_root / pair_name / "tables"
        series_dir = out_root / pair_name / "series"
        tables_dir.mkdir(parents=True, exist_ok=True)
        series_dir.mkdir(parents=True, exist_ok=True)

        for ds_path in dataset_paths:
            stem = Path(ds_path).stem
            prefix = _resolve_prefix(stem)
            table_csv  = tables_dir / f"{prefix}_{stem}_df.csv"
            series_csv = series_dir / f"{prefix}_{stem}_dfs.csv"

            if table_csv.exists() and series_csv.exists():
                n_skip += 1
                continue

            t0 = time.time()
            try:
                table_df, series_df = run_pair_one(
                    h5ad_path=ds_path,
                    pair_name=pair_name,
                    epochs=epochs,
                    silent=True,
                )
                table_df.to_csv(table_csv, index=False)
                series_df.to_csv(series_csv, index=False)
                n_done += 1
                log.info("  ✓ %s / %s (%.0fs)  rows: table=%d, series=%d",
                         pair_name, stem, time.time() - t0, len(table_df), len(series_df))
            except Exception as exc:
                n_err += 1
                log.warning("  ✗ %s / %s failed: %s: %s",
                            pair_name, stem, type(exc).__name__, exc)

    log.info("pair_sweep complete: %d new, %d skipped (resumed), %d errors",
             n_done, n_skip, n_err)


if __name__ == "__main__":
    main()
