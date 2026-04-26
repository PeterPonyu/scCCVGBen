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
PANEL_DPI:    int   = 200       # publication 2x density (still ~ 5 MB / fig)


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
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
