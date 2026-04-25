#!/usr/bin/env python3
"""build_ucb_timeseries.py — Concatenate UCB 10x_mtx timecourse into a single h5ad.

The revised scCCVGBen2_UCB notebook reads 5 separate 10x_mtx directories (CT / D4 /
D7 / D11 / D14) and concatenates them into an AnnData for downstream scCCVGBen
training. This script reproduces that step so STAGE 2 can consume a single
`ucb_timeseries.h5ad` file instead of re-running the combine logic per run.

Inputs (default):
    /home/zeyufu/Desktop/UCBfiles/{CT,D4,D7,D11,D14}/

Output:
    workspace/data/scrna/ucb_timeseries.h5ad

Idempotent: skips if output already exists (unless --force).

Usage:
    python scripts/build_ucb_timeseries.py
    python scripts/build_ucb_timeseries.py --force
    python scripts/build_ucb_timeseries.py --src /path/to/UCBfiles --out workspace/data/scrna/ucb_timeseries.h5ad
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SRC = Path("/home/zeyufu/Desktop/UCBfiles")
DEFAULT_OUT = Path("/home/zeyufu/LAB/scCCVGBen/workspace/data/scrna/ucb_timeseries.h5ad")
STAGES = ("CT", "D4", "D7", "D11", "D14")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if output exists.")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        log.info("Output exists, skipping: %s (--force to rebuild)", args.out)
        return

    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as exc:
        log.error("Install scanpy + anndata: %s", exc)
        sys.exit(1)

    adatas = []
    for stage in STAGES:
        d = args.src / stage
        if not d.is_dir():
            log.warning("Stage dir missing: %s — skipping", d)
            continue
        log.info("Reading %s", d)
        a = sc.read_10x_mtx(str(d))
        a.obs["stage"] = stage
        a.obs["stage_day"] = 0 if stage == "CT" else int(stage.lstrip("D"))
        adatas.append(a)

    if not adatas:
        log.error("No UCB stages found in %s", args.src)
        sys.exit(1)

    log.info("Concatenating %d stages", len(adatas))
    combined = ad.concat(
        adatas,
        axis=0,
        join="outer",
        merge="same",
        label="stage_batch",
        keys=[a.obs["stage"].iloc[0] for a in adatas],
    )
    combined.obs_names_make_unique()
    combined.uns["source_dataset"] = "UCB_timeseries"
    combined.uns["stages"] = list(STAGES)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing %s (%d cells x %d genes)", args.out, combined.n_obs, combined.n_vars)
    combined.write_h5ad(args.out)
    print(f"Built UCB timeseries: {args.out} ({combined.n_obs} cells, {combined.n_vars} genes)")


if __name__ == "__main__":
    main()
