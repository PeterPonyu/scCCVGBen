"""Latent dim × top-K gene mini-UMAP grid.

Each row corresponds to a latent dimension (the ``rows_show`` first dims by
default), each column to a top-correlated gene of that dim. Each cell is a
small UMAP scatter coloured by the relevant gene expression. The first
column of each row is the latent-dim itself for reference.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .panel import render_placeholder
from .scatter import render_continuous_scatter


def render_gene_grid(
    ax_grid: plt.Axes,
    *,
    umap: np.ndarray,
    latent: np.ndarray,
    top_k_df: pd.DataFrame,
    expression: pd.DataFrame,
    rows_show: int = 3,
    cols_show: int = 5,
    title: str = "Latent dim × top-correlated genes",
) -> None:
    """Replace ``ax_grid`` with a (rows × cols) inset grid of mini scatter plots.

    The host ``ax_grid`` is hidden; the figure receives ``rows_show*cols_show``
    new sub-axes via :meth:`Figure.add_subplot` placed inside the bounding
    box of ``ax_grid``.
    """
    if latent is None or top_k_df is None or top_k_df.empty:
        render_placeholder(ax_grid, "no genes")

        return

    fig = ax_grid.figure
    bbox = ax_grid.get_position()
    ax_grid.axis("off")

    ax_grid.set_title(title, pad=4)

    R = min(rows_show, latent.shape[1])
    C = min(cols_show, top_k_df["rank"].max() + 1)
    if R == 0 or C == 0:
        render_placeholder(ax_grid, "empty grid")

        return

    # Reserve top strip of ax_grid for the title (already set above)
    # Carve a tighter inner box leaving small inset
    pad_y = 0.012
    inner_x0 = bbox.x0 + 0.005
    inner_x1 = bbox.x1 - 0.005
    inner_y0 = bbox.y0 + 0.004
    inner_y1 = bbox.y1 - 0.030  # leave headroom for title
    cell_w = (inner_x1 - inner_x0) / C
    cell_h = (inner_y1 - inner_y0) / R

    for r in range(R):
        for c in range(C):
            row_df = top_k_df[(top_k_df["dim"] == r) & (top_k_df["rank"] == c)]
            if row_df.empty:
                continue
            gene = str(row_df.iloc[0]["gene"])
            rho = float(row_df.iloc[0]["rho"])

            ax = fig.add_axes([
                inner_x0 + c * cell_w + 0.002,
                inner_y1 - (r + 1) * cell_h + pad_y / 2,
                cell_w - 0.004,
                cell_h - pad_y,
            ])
            if gene not in expression.columns:
                render_placeholder(ax, "n/a")

                continue
            render_continuous_scatter(
                ax, umap, expression[gene].to_numpy(),
                title=f"d{r}·{gene} ρ{rho:+.2f}",
                cmap="viridis", point_size=2.0, show_colorbar=False,
            )
