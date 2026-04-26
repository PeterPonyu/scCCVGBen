"""UMAP-style scatter primitives for the case figure.

All functions accept an existing :class:`matplotlib.axes.Axes` and a payload
of pre-computed coordinates so the composer keeps full control of the layout
(no implicit ``plt.subplots`` calls).
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .panel import render_placeholder


def _hide_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])

    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_categorical_scatter(
    ax: plt.Axes,
    umap: np.ndarray,
    labels: pd.Series | Sequence[str],
    *,
    title: str = "",
    legend_loc: str | None = "right",
    point_size: float = 4.0,
    palette: str = "tab20",
    max_legend: int = 8,
) -> None:
    """UMAP scatter coloured by a categorical column.

    Legend is rendered inside the axes (loc="best") only when ``len(cats)`` is
    at most ``max_legend``; otherwise a small badge announcing the category
    count is placed in the lower-right so the viewer knows colors are
    meaningful even though no legend is shown. This keeps the figure inside
    its canonical 16x12-inch bounding box (no ``bbox="tight"`` expansion).
    """
    if umap is None or len(umap) == 0:
        render_placeholder(ax, "no UMAP")
        return
    labels = pd.Series(labels).astype(str).reset_index(drop=True)
    cats = list(labels.unique())
    cmap = plt.get_cmap(palette, max(len(cats), 1))
    colors = {c: cmap(i % cmap.N) for i, c in enumerate(cats)}
    for c in cats:
        sel = (labels == c).to_numpy()
        ax.scatter(umap[sel, 0], umap[sel, 1], s=point_size, c=[colors[c]],
                   linewidths=0, alpha=0.85, label=c)
    _hide_axis(ax)
    if title:
        ax.set_title(title, pad=4)
    if legend_loc and len(cats) <= max_legend:
        ax.legend(loc="best", fontsize=6, frameon=False, markerscale=1.5,
                  handlelength=0.6, handletextpad=0.3, labelspacing=0.2)
    elif legend_loc:
        ax.text(
            0.99, 0.02, f"{len(cats)} categories",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="#475569",
            bbox=dict(facecolor="white", edgecolor="#CBD5E1",
                      boxstyle="round,pad=0.18", lw=0.5, alpha=0.85),
        )


def render_continuous_scatter(
    ax: plt.Axes,
    umap: np.ndarray,
    values: np.ndarray | pd.Series,
    *,
    title: str = "",
    cmap: str = "viridis",
    point_size: float = 4.0,
    show_colorbar: bool = True,
) -> None:
    """UMAP scatter coloured by a continuous value (latent dim or gene)."""
    if umap is None or len(umap) == 0:
        render_placeholder(ax, "no UMAP")

        return
    v = np.asarray(values, dtype=float)
    if v.size != umap.shape[0]:
        render_placeholder(ax, "len(values) ≠ N")
        return
    sc = ax.scatter(umap[:, 0], umap[:, 1], c=v, s=point_size,
                    cmap=cmap, linewidths=0, alpha=0.9)
    _hide_axis(ax)
    if title:
        ax.set_title(title, pad=4)
    if show_colorbar:
        cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
