"""Standardized Panel geometry — single source of truth for figure layout.

Every visualize module renders into a :class:`matplotlib.axes.Axes` provided
by the composer. The composer instantiates one :class:`matplotlib.figure.Figure`
per case with a fixed 16x12-inch canvas at 200 DPI and slices it via
``GridSpec`` according to the ``LAYOUT`` defined in ``compose/case_figure.py``.

This module exposes the canonical constants and a :func:`render_placeholder`
that fills an axis with a "missing" stamp so a single failed compute step
does not break the composed figure.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt


PANEL_W_INCH: float = 16.0      # full case figure width
PANEL_H_INCH: float = 16.0      # full case figure height — accommodates 5 rows
# 144 DPI keeps text crisp at submission size while keeping each case PNG
# under ~1 MB. 200 DPI produced 3+ MB PNGs that bloated the stitched
# pair figures (fig11/12/13) past 3 MB each.
PANEL_DPI:    int   = 144


@dataclass(frozen=True)
class PanelSpec:
    """Geometric constants for a single case figure."""
    w_inch: float = PANEL_W_INCH
    h_inch: float = PANEL_H_INCH
    dpi:    int   = PANEL_DPI


def render_placeholder(ax: plt.Axes, message: str = "missing") -> None:
    """Fill ``ax`` with a centred "missing" stamp + bordered frame."""
    ax.set_xlim(0, 1)

    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#94A3B8")
        spine.set_linewidth(0.6)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        fontsize=10, color="#64748B", style="italic",
        transform=ax.transAxes,
    )


def apply_publication_rcparams() -> None:
    """Set matplotlib rcParams for publication consistency.

    Idempotent


    safe to call once per process.
    """
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Critical: never use "tight" bbox here. Some cases (e.g. UCB with
        # many cell-type categories) had legends overflow the canvas, and
        # "tight" expanded the saved figure to ~28× the intended pixel
        # count (200 MP, exceeding the PIL decompression-bomb limit).
        # "standard" honours the figsize the composer set up.
        "savefig.bbox": "standard",
        "savefig.dpi": PANEL_DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
        # Body text bumped from 9pt to 12pt so per-case panels stay readable
        # when stitched into the wider pair figures (fig11/12/13). Pair
        # title is 22pt; body 12pt → ratio < 2x → no visual cliff.
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
