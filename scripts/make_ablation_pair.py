#!/usr/bin/env python3
"""make_ablation_pair.py — render one CCVGAE-style pair ablation figure.

For each ``pair_name`` (Linear_pair / CouVAE_pair / VGAE_pair) generates
``figures/{out_stem}.pdf`` + ``.png`` using the same 20-metric publication
grid as ``scripts/make_axisA_figure.py``. Each panel shows the two
variants (e.g. ``VAE`` vs ``CenVAE``) side-by-side with Wilcoxon
significance brackets, mirroring the legacy CCVGAE
``CG_results/composed_figures/fig{03,04,05}.pdf`` series.

Mapping to CCVGAE figure numbers
================================

  fig05 — Linear_pair  → ``CenVAE vs VAE``
  fig06 — CouVAE_pair  → ``CouVAE vs VAE``
  fig07 — VGAE_pair    → ``GAT-VAE vs VAE``

The legacy CCVGAE pair figures use ``RigorousExperimentalAnalyzer``; we
reuse the project-native :mod:`scccvgben.figures` plumbing instead so the
pipeline does not depend on the deprecated ``REA`` module.

Usage
-----
::

    python scripts/make_ablation_pair.py --pair Linear_pair --out fig05
    python scripts/make_ablation_pair.py --pair CouVAE_pair --out fig06
    python scripts/make_ablation_pair.py --pair VGAE_pair   --out fig07
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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


_PAIR_TO_FOLDER = {
    "Linear_pair": "Linear_pair",
    "CouVAE_pair": "CouVAE_pair",
    "VGAE_pair":   "VGAE_pair",
}
# Fixed left-to-right ordering of the two variants per pair (legacy CCVGAE
# convention: baseline VAE first, "improvement" second). The y-label of the
# significance bracket then reads "improvement vs VAE".
_PAIR_METHOD_ORDER = {
    "Linear_pair": ("VAE", "CenVAE"),
    "CouVAE_pair": ("VAE", "CouVAE"),
    "VGAE_pair":   ("VAE", "GAT-VAE"),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", required=True, choices=list(_PAIR_TO_FOLDER))
    p.add_argument("--tables-root", type=Path,
                   default=REPO_ROOT / "results" / "pair_sweep")
    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT / "data" / "benchmark_manifest.csv")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--out", default=None,
                   help="Output stem (default fig_{pair}_ablation).")
    p.add_argument("--target-n", type=int, default=46,
                   help="Expected dataset count for the pair sweep "
                        "(default 46 = the 45 'new' set + 1 supplement).")
    p.add_argument("--partial-ok", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_publication_rcparams()

    pair_dir = args.tables_root / _PAIR_TO_FOLDER[args.pair] / "tables"
    if not pair_dir.is_dir():
        log.error("pair tables dir missing: %s", pair_dir)
        return 1

    long_df = melt_sweep(pair_dir, modality="scrna", metrics=NUMERIC_METRICS)
    if long_df.empty:
        log.error("no rows from %s", pair_dir)
        return 1

    long_df = filter_to_manifest(long_df, args.manifest, modality="scrna")
    method_order = _PAIR_METHOD_ORDER[args.pair]
    long_df = long_df[long_df["method"].isin(method_order)].copy()

    n_datasets = long_df["dataset_id"].nunique()
    log.info("pair=%s, datasets: %d (target %d)", args.pair, n_datasets, args.target_n)

    if n_datasets < args.target_n and not args.partial_ok:
        log.error("only %d/%d datasets — pass --partial-ok", n_datasets, args.target_n)
        return 1

    metrics = available_numeric_metrics(long_df, NUMERIC_METRICS)
    log.info("metric coverage: %d/%d display metrics", len(metrics), len(NUMERIC_METRICS))

    fig, _ = create_metric_grid_figure(
        long_df,
        metric_grid=METRIC_PANEL_GRID,
        reference_method=method_order[0],   # VAE baseline
        method_order=list(method_order),
        family_titles=METRIC_FAMILY_TITLES,
        title=f"{method_order[1]} vs {method_order[0]} — variant ablation",
        subtitle=f"{n_datasets} scRNA datasets · {len(metrics)} display metrics",
    )

    stem = args.out or f"fig_{args.pair.lower()}_ablation"
    pdf_name = preliminary_path(stem, n_datasets, args.target_n, suffix=".pdf").name
    png_name = preliminary_path(stem, n_datasets, args.target_n, suffix=".png").name
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
