"""Heatmap-style panels: latent self-correlation + tabular gene rankings."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .panel import render_placeholder


def render_latent_corr(ax: plt.Axes, corr: np.ndarray, *, title: str = "Latent self-correlation") -> None:
    """Symmetric (L, L) absolute correlation as a heat map."""
    if corr is None or corr.size == 0:
        render_placeholder(ax, "no latent corr")

        return
    im = ax.imshow(corr, cmap="magma_r", vmin=0, vmax=1, aspect="equal")
    L = corr.shape[0]
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(range(L), fontsize=6)
    ax.set_yticklabels(range(L), fontsize=6)
    ax.set_xlabel("latent dim", fontsize=7)
    ax.set_ylabel("latent dim", fontsize=7)
    ax.set_title(title, pad=4)
    # Use a small inset axis for the colorbar so the heatmap fills the panel
    # without the cbar consuming ~half the panel width.
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.038, shrink=0.85, pad=0.012)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("|corr|", fontsize=6)


def render_top_gene_table(ax: plt.Axes, top_k_df: pd.DataFrame,
                          *, title: str = "Top-1 correlated gene per latent dim") -> None:
    """Compact text table — one row per latent dim, top 1 gene shown."""
    if top_k_df is None or top_k_df.empty:
        render_placeholder(ax, "no genes")

        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, pad=4)

    # Two-column layout: 5 dims per column. Fits the half-height row 3 of
    # the case figure without wasted whitespace.
    rank0 = top_k_df[top_k_df["rank"] == 0].sort_values("dim")
    rows = [(int(r["dim"]), str(r["gene"]), float(r["rho"])) for _, r in rank0.iterrows()]
    n = len(rows)
    if n == 0:
        render_placeholder(ax, "no rank-0 rows")
        return

    n_cols = 2
    per_col = (n + n_cols - 1) // n_cols
    y_top = 0.92
    y_bot = 0.06
    y_step = (y_top - y_bot) / max(per_col, 1)
    col_w = 0.5
    for col_idx in range(n_cols):
        x0 = col_idx * col_w
        # column header
        ax.text(x0 + 0.02, 0.98, "dim",  fontsize=7.5, fontweight="bold")
        ax.text(x0 + 0.10, 0.98, "gene", fontsize=7.5, fontweight="bold")
        ax.text(x0 + 0.42, 0.98, "ρ",    fontsize=7.5, fontweight="bold")
        for j in range(per_col):
            idx = col_idx * per_col + j
            if idx >= n:
                break
            d, g, rho = rows[idx]
            y = y_top - (j + 0.5) * y_step
            color = "#0EA5E9" if rho > 0 else "#EF4444"
            ax.text(x0 + 0.02, y, str(d),       fontsize=7)
            ax.text(x0 + 0.10, y, g[:24],       fontsize=7, fontfamily="monospace")
            ax.text(x0 + 0.42, y, f"{rho:+.3f}", fontsize=7, color=color)
