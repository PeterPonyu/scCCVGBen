"""Per-condition latent-value violin plot.

Shows the distribution of each latent dimension stratified by the case's
biological condition. Helps detect dimensions that capture condition-level
biology (e.g. young vs aged, tumor vs normal) versus those that capture
within-condition heterogeneity.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .panel import render_placeholder


def render_condition_violin(
    ax: plt.Axes,
    latent: np.ndarray,
    condition: pd.Series,
    *,
    n_dims_show: int = 5,
    title: str = "Latent values by condition",
    palette: str = "Set2",
) -> None:
    """Vertical violin matrix: ``n_dims_show`` latent dims × condition groups."""
    if latent is None or latent.size == 0:
        render_placeholder(ax, "no latent")

        return
    cond = pd.Series(condition).astype(str).reset_index(drop=True)
    cats = list(cond.unique())
    if len(cats) < 2:
        # Degenerate condition — fall back to a single boxplot per dim
        cats = ["all"]
    L = min(n_dims_show, latent.shape[1])
    cmap = plt.get_cmap(palette, max(len(cats), 2))

    width = 0.8 / max(len(cats), 1)
    positions = np.arange(L)
    for i, c in enumerate(cats):
        sel = (cond == c).to_numpy() if c != "all" else np.ones(len(cond), dtype=bool)
        if sel.sum() == 0:
            continue
        data = [latent[sel, d] for d in range(L)]
        offsets = positions + (i - (len(cats) - 1) / 2) * width
        parts = ax.violinplot(data, positions=offsets, widths=width * 0.95,
                              showextrema=False, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(cmap(i % cmap.N))
            pc.set_edgecolor("#1F2D3D")
            pc.set_alpha(0.85)
            pc.set_linewidth(0.4)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#1F2D3D")
            parts["cmedians"].set_linewidth(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"d{d}" for d in range(L)], fontsize=9)
    ax.set_ylabel("latent value", fontsize=10)
    ax.set_title(title, pad=4, fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    # Legend: only when condition count fits inside the panel; otherwise show
    # a small badge with the category count to keep the panel within its
    # canonical bounding box (no canvas expansion).
    MAX_LEGEND = 8
    if len(cats) <= MAX_LEGEND:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=cmap(i % cmap.N), label=c) for i, c in enumerate(cats)]
        ax.legend(handles=handles, loc="best", fontsize=8, frameon=False,
                  handlelength=1.0, handletextpad=0.3, labelspacing=0.2)
    else:
        ax.text(
            0.99, 0.02, f"{len(cats)} categories",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#475569",
            bbox=dict(facecolor="white", edgecolor="#CBD5E1",
                      boxstyle="round,pad=0.18", lw=0.5, alpha=0.85),
        )
