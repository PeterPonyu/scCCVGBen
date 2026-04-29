"""UMAP-style scatter primitives for the case figure.

All functions accept an existing :class:`matplotlib.axes.Axes` and a payload
of pre-computed coordinates so the composer keeps full control of the layout
(no implicit ``plt.subplots`` calls).
"""
from __future__ import annotations

import re
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .panel import render_placeholder


def _hide_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])

    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _short_category_label(label: object, *, max_chars: int = 16) -> str:
    text = str(label).strip()
    sample_match = re.search(r"(?:^|[_\W])sample[_\W]*(\d+)\b", text, flags=re.IGNORECASE)
    if sample_match:
        return f"s{sample_match.group(1)}"
    gsm_match = re.match(r"^(GSM\d+)(?:\b|[_\W])", text, flags=re.IGNORECASE)
    if gsm_match:
        return gsm_match.group(1).upper()
    text = re.sub(r"\.(?:csv|tsv|txt|h5ad)(?:\.gz)?$", "", text, flags=re.IGNORECASE)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip(" _-.,;") + "…"


def _reserve_right_gutter(ax: plt.Axes, *, fraction: float = 0.42) -> None:
    """Create empty data-space on the right for legends/lists."""
    x0, x1 = ax.get_xlim()
    span = max(abs(x1 - x0), 1e-9)
    ax.set_xlim(x0, x1 + span * fraction)


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
    summary_limit: int = 8,
    legend_title: str | None = None,
) -> None:
    """UMAP scatter coloured by a categorical column.

    When the category count is modest, a compact legend is rendered in a
    reserved right gutter. For crowded annotations, the same gutter shows the
    most abundant labels plus a ``+N more`` note instead of a vague category
    count. This keeps the figure inside its canonical 16x12-inch bounding box
    (no ``bbox="tight"`` expansion).
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
                   linewidths=0, alpha=0.85)
    if legend_loc == "right":
        _reserve_right_gutter(ax)
    _hide_axis(ax)
    if title:
        ax.set_title(title, pad=4)
    if legend_loc and len(cats) <= max_legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=colors[c],
                   label=_short_category_label(c), markersize=4.5)
            for c in cats
        ]
        ax.legend(
            handles=handles,
            title=legend_title,
            loc="center right" if legend_loc == "right" else "best",
            bbox_to_anchor=(0.995, 0.50) if legend_loc == "right" else None,
            fontsize=7.2,
            title_fontsize=7.5,
            frameon=True,
            facecolor="white",
            edgecolor="#CBD5E1",
            framealpha=0.82,
            handlelength=0.7,
            handletextpad=0.32,
            labelspacing=0.18,
            borderpad=0.24,
        )
    elif legend_loc:
        counts = labels.value_counts()
        shown = list(counts.index[:summary_limit])
        y = 0.95
        if legend_title:
            ax.text(0.735, y, legend_title, transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.8, fontweight="bold",
                    color="#334155")
            y -= 0.075
        for cat in shown:
            ax.text(0.735, y, "●", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color=colors[cat])
            ax.text(
                0.77, y,
                _short_category_label(cat),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.1,
                color="#334155",
            )
            y -= 0.064
        if len(cats) > len(shown):
            ax.text(0.735, y, f"+{len(cats) - len(shown)} more",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.1, color="#64748B")


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
    """UMAP scatter coloured by a continuous value (latent coordinate or gene)."""
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
        cbar.ax.tick_params(labelsize=8)
