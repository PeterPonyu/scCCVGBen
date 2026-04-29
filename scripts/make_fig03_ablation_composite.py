#!/usr/bin/env python3
"""make_fig03_ablation_composite.py — merged ablation-pair composite.

Renders a single figure stacking three super-blocks A/B/C, each a 2 row × 10
col metric grid (full 20 NUMERIC_METRICS) for one ablation pair:

  A — Linear_pair  (CenVAE  vs VAE)
  B — CouVAE_pair  (CouVAE  vs VAE)
  C — VGAE_pair    (GAT-VAE vs VAE)

Each block uses the shared :func:`create_metric_grid_figure` plumbing — same
boxplot + strip overlay, family-coloured panel headers, Wilcoxon Holm-corrected
significance brackets vs the VAE baseline. The super-block letter (A/B/C) is
rendered as a plain bold black panel label at the top-left of each block,
matching the panel-label convention used by every other multi-panel figure.

Output: ``figures/fig03.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scccvgben.figures import (  # noqa: E402
    METRIC_FAMILY_TITLES,
    METRIC_LABELS,
    METRIC_PANEL_GRID,
    NUMERIC_METRICS,
    apply_publication_rcparams,
    create_metric_grid_figure,
    filter_to_manifest,
    metric_coverage_audit,
    palette_for_methods,
    preliminary_path,
    write_metric_audit,
)
from scccvgben.figures._long_form import melt_sweep  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

_BLOCKS: tuple[tuple[str, str, tuple[str, str]], ...] = (
    ("A", "Linear_pair", ("VAE", "CenVAE")),
    ("B", "CouVAE_pair", ("VAE", "CouVAE")),
    ("C", "VGAE_pair",   ("VAE", "GAT-VAE")),
)

_VARIANT_LABELS = {
    "VAE": "Base",
    "CenVAE": "Centroid",
    "CouVAE": "Coupling",
    "GAT-VAE": "GAT",
}

# 2-row × 10-col reshape of the canonical 4-row × 5-col METRIC_PANEL_GRID.
_GRID_2x10: tuple[tuple[str, ...], ...] = (
    METRIC_PANEL_GRID[0] + METRIC_PANEL_GRID[1],
    METRIC_PANEL_GRID[2] + METRIC_PANEL_GRID[3],
)

_BLOCK_GAP_PX = 18        # vertical gap between blocks when stitching
_TARGET_WIDTH_PX = 4800   # shared block image width (matplotlib renders vary)
_DPI = 300


def _scale_to_width(img: Image.Image, target_w: int) -> Image.Image:
    if img.width == target_w:
        return img
    new_h = max(1, round(img.height * target_w / img.width))
    return img.resize((target_w, new_h), Image.Resampling.LANCZOS)


def _render_block(
    long_df: pd.DataFrame,
    method_order: tuple[str, str],
    pair_label: str,
    target_n: int,
    out_png: Path,
) -> None:
    """Render one pair-block (2 rows × 10 cols) with a single A/B/C label."""
    long_df = long_df[long_df["method"].isin(method_order)].copy()
    long_df["variant_display"] = long_df["method"].map(_VARIANT_LABELS).fillna(long_df["method"])
    display_order = [_VARIANT_LABELS.get(method, method) for method in method_order]
    n_datasets = long_df["dataset_id"].nunique()

    fig, _ = create_metric_grid_figure(
        long_df,
        metric_grid=_GRID_2x10,
        group_col="variant_display",
        reference_method=display_order[0],
        method_order=display_order,
        family_titles=METRIC_FAMILY_TITLES,
        # Keep block context in the manuscript caption rather than embedding a
        # repeated title/subtitle/legend band in every stitched block.  The
        # extra vertical room keeps significance brackets and markers clear of
        # headers in Figure 3's tight 2×10 layout.
        title=None,
        subtitle=None,
        metric_labels=METRIC_LABELS,
        per_col_width=2.36,
        per_row_height=3.34,
        xtick_labelsize=20.4,
        ytick_labelsize=14.2,
        xtick_rotation=30,
        hspace=0.42,
        palette=palette_for_methods(display_order),
        panel_label_letter=pair_label,
        significance_marker_fontsize=12.4,
        significance_ns_fontsize=10.2,
        significance_step_fraction=0.150,
        significance_text_pad_fraction=0.07,
    )
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    log.info("block %s: pair=%s, datasets=%d (target %d)",
             pair_label, method_order, n_datasets, target_n)


def _stack_blocks(
    block_pngs: list[Path],
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Vertically stack block PNGs onto a single canvas — no rails or overlays.

    Each block image already carries its own A/B/C panel letter (drawn by
    ``create_metric_grid_figure`` via ``panel_label_letter``).
    """
    blocks = [Image.open(p).convert("RGB") for p in block_pngs]
    blocks = [_scale_to_width(b, _TARGET_WIDTH_PX) for b in blocks]

    width = _TARGET_WIDTH_PX
    height = sum(b.height for b in blocks) + _BLOCK_GAP_PX * (len(blocks) - 1)
    canvas = Image.new("RGB", (width, height), "white")

    y = 0
    for block in blocks:
        canvas.paste(block, (0, y))
        y += block.height + _BLOCK_GAP_PX

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, dpi=(_DPI, _DPI))
    canvas.save(out_pdf, "PDF", resolution=_DPI)


def _audit_for_block(
    long_df: pd.DataFrame,
    method_order: tuple[str, str],
    expected_n: int,
) -> pd.DataFrame:
    sub = long_df[long_df["method"].isin(method_order)].copy()
    return metric_coverage_audit(
        sub, NUMERIC_METRICS, figure_id="fig03",
        group_col="method", expected_datasets=expected_n,
        expected_methods=len(method_order),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-root", type=Path,
                        default=REPO_ROOT / "results" / "pair_sweep")
    parser.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "data" / "benchmark_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--out", default="fig03_ablation_composite",
                        help="Output stem (default fig03_ablation_composite).")
    parser.add_argument("--target-n", type=int, default=100,
                        help="Expected dataset count per pair (default 100).")
    parser.add_argument("--partial-ok", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_publication_rcparams()

    block_pngs: list[Path] = []
    audit_frames: list[pd.DataFrame] = []
    dataset_counts: list[int] = []

    tmp_dir = Path(tempfile.mkdtemp(prefix="fig03_"))
    for letter, sub_name, method_order in _BLOCKS:
        pair_dir = args.tables_root / sub_name / "tables"
        if not pair_dir.is_dir():
            log.error("missing pair tables: %s", pair_dir)
            return 1
        long_df = melt_sweep(pair_dir, modality="scrna", metrics=NUMERIC_METRICS)
        long_df = filter_to_manifest(long_df, args.manifest, modality="scrna")
        if long_df.empty:
            log.error("no rows in %s", pair_dir)
            return 1
        n_datasets = long_df["dataset_id"].nunique()
        dataset_counts.append(n_datasets)
        if n_datasets < args.target_n and not args.partial_ok:
            log.error("only %d/%d datasets in %s — pass --partial-ok",
                      n_datasets, args.target_n, sub_name)
            return 1

        block_png = tmp_dir / f"block_{letter}.png"
        _render_block(long_df, method_order, letter, args.target_n, block_png)
        block_pngs.append(block_png)
        audit_frames.append(_audit_for_block(long_df, method_order, args.target_n))

    n_min = min(dataset_counts)
    out_pdf = args.out_dir / preliminary_path(
        args.out, n_min, args.target_n, suffix=".pdf").name
    out_png = args.out_dir / preliminary_path(
        args.out, n_min, args.target_n, suffix=".png").name

    _stack_blocks(block_pngs, out_png, out_pdf)
    log.info("wrote %s", out_pdf)
    log.info("wrote %s", out_png)

    audit = pd.concat(audit_frames, ignore_index=True)
    write_metric_audit(args.out_dir / "_metric_audit.csv", audit, figure_id="fig03")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
