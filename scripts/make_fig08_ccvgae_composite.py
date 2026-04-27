#!/usr/bin/env python3
"""make_fig08_ccvgae_composite.py — CCVGAE flagship vs the three pair variants.

Combines the per-pair tables under ``results/pair_sweep/`` with the
scCCVGBen flagship row (``scCCVGBen_GAT`` from
``results/encoder_sweep/{ds}.csv``) so a single 20-metric grid shows
**five methods side-by-side**: ``VAE``, ``CenVAE``, ``CouVAE``,
``GAT-VAE``, ``CCVGAE``. This mirrors the legacy CCVGAE composite figures
``fig06`` (CCVGAE vs CenVAE) + ``fig07`` (CCVGAE vs CouVAE) folded into a
single comparison.

Output: ``figures/fig08.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scccvgben.figures import (  # noqa: E402
    METRIC_FAMILY_TITLES,
    METRIC_PANEL_GRID,
    NUMERIC_METRICS,
    apply_publication_rcparams,
    available_numeric_metrics,
    create_metric_grid_figure,
    filter_to_manifest,
    preliminary_path,
)
from scccvgben.figures._long_form import melt_sweep  # noqa: E402

log = logging.getLogger(__name__)


_METHOD_ORDER = ("VAE", "CenVAE", "CouVAE", "GAT-VAE", "CCVGAE")
# Map the encoder-sweep label to the public 'CCVGAE' alias used in the
# paper: scCCVGBen_GAT is the flagship configuration.
_FLAGSHIP_SOURCE_METHOD = "scCCVGBen_GAT"
_FLAGSHIP_DISPLAY_NAME  = "CCVGAE"


def _load_flagship(encoder_dir: Path, manifest: Path) -> pd.DataFrame:
    """Pull the ``scCCVGBen_GAT`` row from each encoder-sweep CSV, relabel as
    ``CCVGAE``, melt to long form."""
    df = melt_sweep(encoder_dir, modality="scrna", metrics=NUMERIC_METRICS)
    df = filter_to_manifest(df, manifest, modality="scrna")
    df = df[df["method"] == _FLAGSHIP_SOURCE_METHOD].copy()
    df["method"] = _FLAGSHIP_DISPLAY_NAME
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-root", type=Path,
                   default=REPO_ROOT / "results" / "pair_sweep")
    p.add_argument("--encoder-dir", type=Path,
                   default=REPO_ROOT / "results" / "encoder_sweep_flagship",
                   help="Source for the CCVGAE flagship row (default: "
                        "encoder_sweep_flagship/, which trains scCCVGBen_GAT "
                        "at epochs=200, i_dim=10 — matching the pair_sweep "
                        "ablation columns. Pass --encoder-dir results/"
                        "encoder_sweep to fall back to the 100-epoch source.")
    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT / "data" / "benchmark_manifest.csv")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--out", default="fig08")
    p.add_argument("--target-n", type=int, default=100,
                   help="100 = scccvgben benchmark_manifest scRNA size "
                        "(45 CCVGAE-overlap reused via reuse_ccvgae_pairs.py + "
                        "55 NEW from run_pair_sweep.py).")
    p.add_argument("--partial-ok", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_publication_rcparams()

    frames = []
    # Pair tables — only keep rows whose method is in the canonical order
    keep = set(_METHOD_ORDER) - {_FLAGSHIP_DISPLAY_NAME}
    for sub in ("Linear_pair", "CouVAE_pair", "VGAE_pair"):
        d = args.pair_root / sub / "tables"
        if not d.is_dir():
            log.warning("missing pair dir: %s", d)
            continue
        df = melt_sweep(d, modality="scrna", metrics=NUMERIC_METRICS)
        df = filter_to_manifest(df, args.manifest, modality="scrna")
        frames.append(df[df["method"].isin(keep)])

    if not frames:
        log.error("no pair tables found under %s", args.pair_root)
        return 1
    long_pair = pd.concat(frames, ignore_index=True)
    # Each (pair_name, method=VAE) row will appear ~3 times — collapse to one
    # observation per (dataset_id, method, metric) by averaging (gives the
    # cross-pair pooled VAE estimate the figure expects).
    long_pair = (
        long_pair.groupby(["dataset_id", "method", "metric", "modality"], as_index=False)
                 ["value"].mean()
    )

    long_flag = _load_flagship(args.encoder_dir, args.manifest)
    if long_flag.empty:
        log.error("no flagship %s rows in %s", _FLAGSHIP_SOURCE_METHOD, args.encoder_dir)
        return 1

    cols = ["dataset_id", "dataset_name", "modality", "method", "metric", "value"]
    long_pair["dataset_name"] = long_pair["dataset_id"]
    pair_cols = [c for c in cols if c in long_pair.columns]
    flag_cols = [c for c in cols if c in long_flag.columns]
    long_df = pd.concat(
        [long_pair[pair_cols], long_flag[flag_cols]],
        ignore_index=True,
        join="outer",
    )

    n_datasets = long_df["dataset_id"].nunique()
    methods_present = [m for m in _METHOD_ORDER if m in long_df["method"].unique()]
    log.info("methods: %s", methods_present)
    log.info("datasets: %d (target %d)", n_datasets, args.target_n)

    if n_datasets < args.target_n and not args.partial_ok:
        log.error("only %d/%d datasets — pass --partial-ok", n_datasets, args.target_n)
        return 1

    metrics = available_numeric_metrics(long_df, NUMERIC_METRICS)
    log.info("metric coverage: %d/%d display metrics", len(metrics), len(NUMERIC_METRICS))

    fig, _ = create_metric_grid_figure(
        long_df,
        metric_grid=METRIC_PANEL_GRID,
        reference_method=_FLAGSHIP_DISPLAY_NAME,
        method_order=methods_present,
        family_titles=METRIC_FAMILY_TITLES,
        title="CCVGAE flagship vs CenVAE / CouVAE / GAT-VAE / VAE",
        subtitle=f"{n_datasets} scRNA datasets · {len(metrics)} display metrics",
    )

    pdf_name = preliminary_path(args.out, n_datasets, args.target_n, suffix=".pdf").name
    png_name = preliminary_path(args.out, n_datasets, args.target_n, suffix=".png").name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = args.out_dir / pdf_name
    png_out = args.out_dir / png_name
    fig.savefig(pdf_out, bbox_inches="tight")
    fig.savefig(png_out, bbox_inches="tight", dpi=200)
    log.info("wrote %s", pdf_out)
    log.info("wrote %s", png_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
