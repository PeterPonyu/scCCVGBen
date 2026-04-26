"""Single-case figure composer.

Layout (16in × 12in, 200 DPI)::

    ┌───────────┬───────────┬───────────┬─────────────┐
    │ A         │ B         │ C         │ D           │
    │ UMAP      │ UMAP      │ UMAP      │ Latent      │
    │ condition │ cell type │ latent d0 │ self-corr   │
    ├───────────┴───────────┴───────────┼─────────────┤
    │ F                                 │ G           │
    │ Latent dim × top-K gene grid      │ Condition   │
    │ (3 dims × 5 genes mini-UMAPs)     │ violin      │
    │                                   │             │
    ├───────────────────────────────────┴─────────────┤
    │ E   Top-1 correlated gene per latent dim table  │
    └─────────────────────────────────────────────────┘

The composer never raises on missing panels — every cell is wrapped in
try/except and falls back to a placeholder.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ..visualize.panel import (
    PANEL_W_INCH, PANEL_H_INCH, PANEL_DPI,
    apply_publication_rcparams, render_placeholder,
)
from ..visualize.scatter import render_categorical_scatter, render_continuous_scatter
from ..visualize.heatmap import render_latent_corr, render_top_gene_table
from ..visualize.violin import render_condition_violin
from ..visualize.gene_grid import render_gene_grid

log = logging.getLogger(__name__)


def _safe(fn, ax, *args, **kw) -> None:
    """Call ``fn(ax, *args, **kw)`` and on any exception fall back to a placeholder."""
    try:
        fn(ax, *args, **kw)
    except Exception as exc:
        log.warning("panel render failed: %s: %s", type(exc).__name__, exc)
        render_placeholder(ax, f"render error: {type(exc).__name__}")


def compose_case_figure(payload: dict, out_dir: Path) -> Path:
    """Render the multi-panel figure for one case payload and save PDF + PNG.

    Returns the PDF path.
    """
    apply_publication_rcparams()
    case = payload["case"]
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(PANEL_W_INCH, PANEL_H_INCH), dpi=PANEL_DPI)
    # height_ratios: row0 (UMAPs+heatmap, square-ish) is shorter than the
    # mid-rows (F gene grid is dense), and row3 (table E) is half-height
    # since the table stays compact regardless of dim count.
    gs = GridSpec(
        nrows=4, ncols=4,
        figure=fig,
        height_ratios=[2.6, 3.2, 3.2, 1.6],
        hspace=0.55, wspace=0.34,
        left=0.045, right=0.975, top=0.91, bottom=0.045,
    )

    # ── Header / title bar ───────────────────────────────────────────
    fig.suptitle(
        f"Bio-validation case · {case.title}",
        fontsize=14, fontweight="bold", y=0.985,
    )
    sub = (
        f"{case.accession} · theme={case.theme} · encoder={case.encoder} · "
        f"N={payload.get('n_obs_subsampled','?')} of {payload.get('n_obs','?')} cells, "
        f"{payload.get('n_vars_post_hvg','?')} HVGs"
    )
    fig.text(0.04, 0.955, sub, fontsize=8.5, color="#475569")

    # ── Row 0 — A B C D ─────────────────────────────────────────────
    ax_A = fig.add_subplot(gs[0, 0])

    _safe(
        render_categorical_scatter, ax_A,
        payload.get("umap"), payload.get("condition"),
        title="A · UMAP coloured by condition", legend_loc="right",
    )
    ax_B = fig.add_subplot(gs[0, 1])

    _safe(
        render_categorical_scatter, ax_B,
        payload.get("umap"), payload.get("cell_type"),
        title="B · UMAP coloured by cell type", legend_loc="right",
    )
    ax_C = fig.add_subplot(gs[0, 2])
    latent = payload.get("latent")
    if latent is not None and latent.shape[1] > 0:
        _safe(
            render_continuous_scatter, ax_C,
            payload.get("umap"), latent[:, 0],
            title="C · UMAP coloured by latent dim 0", cmap="coolwarm",
        )
    else:
        render_placeholder(ax_C, "no latent")
    ax_D = fig.add_subplot(gs[0, 3])

    _safe(
        render_latent_corr, ax_D, payload.get("latent_corr"),
        title="D · |corr| between latent dims",
    )

    # ── Mid rows — F (gene grid) + G (violin) ───────────────────────
    ax_F = fig.add_subplot(gs[1:3, 0:3])
    _safe(
        render_gene_grid, ax_F,
        umap=payload.get("umap"),
        latent=payload.get("latent"),
        top_k_df=payload.get("top_k_genes_df"),
        expression=payload.get("expression"),
        rows_show=3, cols_show=5,
        title="F · Latent dim × top-5 correlated gene mini-UMAPs",
    )
    ax_G = fig.add_subplot(gs[1:3, 3])
    _safe(
        render_condition_violin, ax_G,
        payload.get("latent"), payload.get("condition"),
        n_dims_show=5,
        title="G · Per-condition latent value distribution (first 5 dims)",
    )

    # ── Bottom row — E (table) ──────────────────────────────────────
    ax_E = fig.add_subplot(gs[3, :])
    _safe(
        render_top_gene_table, ax_E, payload.get("top_k_genes_df"),
        title="E · Top-1 correlated gene per latent dim",
    )

    pdf_path = out_dir / f"fig_biovalidation_case_{case.case_id}.pdf"
    png_path = out_dir / f"fig_biovalidation_case_{case.case_id}.png"
    # Explicit bbox_inches=None forces matplotlib to use the figure's exact
    # configured size (no auto-expansion), which is the bug that produced the
    # 200-megapixel UCB.png on a previous run when bbox="tight" expanded the
    # canvas to fit overflowing legends.
    fig.savefig(pdf_path, bbox_inches=None)
    fig.savefig(png_path, dpi=PANEL_DPI, bbox_inches=None)
    plt.close(fig)
    log.info("[compose] %s + .png", pdf_path)
    return pdf_path
