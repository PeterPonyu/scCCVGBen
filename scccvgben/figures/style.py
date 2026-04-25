"""DataFrame-native publication figure helper.

Visual contract reproduced from the read-only reference implementation
WITHOUT importing its analyzer module or coupling to its state machine.
All figures consume a pre-melted long DataFrame.

Visual contract:
- box + strip overlay (seaborn).
- Wilcoxon Holm-corrected significance brackets (<= 3 per panel).
- Panel labels A/B/C... at panel_label_position.
- Font fallback Arial -> Liberation Sans, pdf.fonttype = 42, 300 DPI.
- Spectral palette default.

State isolation (binding, per Architect refinement #1):
- create_publication_figure wraps render in plt.rc_context, after
  mpl.rcdefaults() to discard whatever a sibling figure module may have
  set globally (for example module-level style and palette mutations).
- seaborn palette is reset inside the same scope.
- Backend pinned to Agg at module import.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

from ._significance import select_significance_pairs
from .metrics import METRIC_PANEL_GRID, METRIC_TO_FAMILY

PUBLICATION_RCPARAMS: dict = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.dpi": 100,
}


def apply_publication_rcparams() -> None:
    """Reset rcParams to defaults, then apply the publication contract.

    Idempotent. Call at the top of any figure script before plotting.
    Unlike additive rcParams setters, this clears modifications a sibling
    module may have made (for example module-level style mutations).
    """
    mpl.rcdefaults()
    mpl.rcParams.update(PUBLICATION_RCPARAMS)
    sns.set_palette("Spectral")
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def _draw_significance_brackets(
    ax: plt.Axes,
    pairs: list[tuple[str, str, float]],
    method_order: Sequence[str],
    y_data_max: float,
) -> None:
    if not pairs:
        return
    positions = {m: i for i, m in enumerate(method_order)}
    span = max(y_data_max, 1e-9)
    step = span * 0.08
    base = y_data_max + step
    for j, (a, b, p) in enumerate(pairs):
        if a not in positions or b not in positions:
            continue
        x1, x2 = positions[a], positions[b]
        y = base + j * step
        ax.plot([x1, x1, x2, x2], [y - step * 0.2, y, y, y - step * 0.2],
                lw=1.0, color="black")
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"
        ax.text((x1 + x2) / 2, y + step * 0.05, sig,
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(top=base + (len(pairs) + 1) * step)


def create_publication_figure(
    long_df: pd.DataFrame,
    metrics: Sequence[str],
    group_col: str = "method",
    *,
    reference_method: str | None = None,
    method_order: Sequence[str] | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
    palette: str = "Spectral",
    panel_label_position: tuple[float, float] = (-0.15, 1.05),
    pdf_fonttype: int = 42,
    pair_col: str = "dataset_id",
    show_significance: bool = True,
    box_width: float = 0.6,
    strip_size: float = 2.5,
    strip_alpha: float = 0.55,
    metric_labels: Mapping[str, str] | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Multi-panel box + strip figure with optional Wilcoxon brackets.

    Forbidden: this function MUST NOT depend on an analyzer instance,
    external state machine, or any glob-based CSV folder ingest. Inputs are a
    long DataFrame and a metric list.
    """
    n = len(metrics)
    if n == 0:
        raise ValueError("metrics must be non-empty")

    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (5.5 * ncols, 4.0 * nrows)
    rcparams = {**PUBLICATION_RCPARAMS, "pdf.fonttype": pdf_fonttype, "savefig.dpi": dpi}

    if method_order is None:
        method_order = (
            long_df[group_col].dropna().unique().tolist()
        )

    panel_letters = [chr(ord("A") + i) for i in range(n)]

    with plt.rc_context(rcparams):
        sns.set_palette(palette)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
        flat_axes: list[plt.Axes] = list(np.atleast_1d(axes).ravel())

        for idx, metric in enumerate(metrics):
            ax = flat_axes[idx]
            sub = long_df.loc[long_df["metric"] == metric].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            sub[group_col] = pd.Categorical(
                sub[group_col], categories=list(method_order), ordered=True
            )
            sub = sub.dropna(subset=[group_col, "value"])

            sns.boxplot(
                data=sub, x=group_col, y="value", order=method_order,
                hue=group_col, hue_order=method_order, legend=False,
                ax=ax, showfliers=False, width=box_width,
                boxprops={"alpha": 0.65}, palette=palette,
            )
            sns.stripplot(
                data=sub, x=group_col, y="value", order=method_order,
                ax=ax, size=strip_size, alpha=strip_alpha, color="black",
                jitter=0.18,
            )
            ax.set_title(metric_labels.get(metric, metric) if metric_labels else metric)
            ax.set_xlabel("")
            ax.set_ylabel("value")
            ax.tick_params(axis="x", rotation=30)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")

            ax.text(
                *panel_label_position, panel_letters[idx],
                transform=ax.transAxes, fontsize=16, fontweight="bold",
                va="top", ha="right",
            )

            if show_significance and reference_method is not None:
                pairs = select_significance_pairs(
                    long_df, metric=metric,
                    reference_method=reference_method,
                    group_col=group_col, pair_col=pair_col,
                    top_k=3,
                )
                if pairs:
                    y_max = float(sub["value"].max())
                    _draw_significance_brackets(ax, pairs, method_order, y_max)

        for j in range(n, len(flat_axes)):
            flat_axes[j].set_visible(False)

        fig.tight_layout()

    return fig, flat_axes[:n]


def _metric_values(
    long_df: pd.DataFrame,
    metric: str,
    group_col: str,
    method_order: Sequence[str],
) -> pd.DataFrame:
    sub = long_df.loc[long_df["metric"] == metric].copy()
    if sub.empty:
        return sub
    sub[group_col] = pd.Categorical(sub[group_col], categories=list(method_order), ordered=True)
    return sub.dropna(subset=[group_col, "value"])


def _label_axis(
    ax: plt.Axes,
    family: str,
    title: str,
    color: str,
) -> None:
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.08, 0.04),
            0.84,
            0.92,
            boxstyle="round,pad=0.025,rounding_size=0.04",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            alpha=0.95,
        )
    )
    family_label = family.replace("-", "\n")
    family_fontsize = 18 if "\n" in family_label else 22
    ax.text(
        0.50,
        0.62,
        family_label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=family_fontsize,
        fontweight="bold",
        color="white",
        linespacing=0.92,
    )
    ax.text(
        0.50,
        0.36,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
        linespacing=1.08,
    )


def create_metric_family_figure(
    long_df: pd.DataFrame,
    metric_families: Sequence[tuple[str, Sequence[str]]],
    group_col: str = "method",
    *,
    reference_method: str | None = None,
    method_order: Sequence[str] | None = None,
    family_titles: Mapping[str, str] | None = None,
    family_colors: Mapping[str, str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    dpi: int = 300,
    palette: str = "Spectral",
    pair_col: str = "dataset_id",
    show_significance: bool = True,
    metric_labels: Mapping[str, str] | None = None,
    max_cols: int | None = None,
    per_col_width: float = 2.72,
    per_row_height: float = 3.18,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Render BEN/DRE/LSE metric families as labelled publication rows.

    The legacy helper above is intentionally compact and flat.  This renderer
    keeps every numeric metric visible while making the metric family structure
    explicit: one row per family, one panel per metric, and a large colour-coded
    family rail.  It remains DataFrame-native and does not recompute results.
    """
    if not metric_families:
        raise ValueError("metric_families must be non-empty")

    if method_order is None:
        method_order = long_df[group_col].dropna().unique().tolist()
    method_order = list(method_order)
    if not method_order:
        raise ValueError("method_order must be non-empty")

    family_titles = family_titles or {}
    default_colors = {
        "BEN": "#1f5f9f",
        "DRE-UMAP": "#6C3483",
        "DRE-tSNE": "#7A4EAB",
        "LSE": "#138D75",
    }
    family_colors = {**default_colors, **(family_colors or {})}
    max_metrics = max(len(metrics) for _, metrics in metric_families)
    ncols = max_cols or max_metrics
    nrows = len(metric_families)

    fig_w = 1.25 + ncols * per_col_width
    fig_h = 1.25 + nrows * per_row_height
    rcparams = {**PUBLICATION_RCPARAMS, "savefig.dpi": dpi, "figure.dpi": dpi}

    with plt.rc_context(rcparams):
        sns.set_palette(palette)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        outer_grid = fig.add_gridspec(
            nrows,
            1,
            left=0.025,
            right=0.992,
            bottom=0.055,
            top=0.90 if title else 0.96,
            hspace=0.52,
        )

        axes: list[plt.Axes] = []
        panel_idx = 0
        for row_idx, (family, metrics) in enumerate(metric_families):
            row_grid = outer_grid[row_idx, 0].subgridspec(
                1,
                len(metrics) + 1,
                width_ratios=[0.62, *([1.0] * len(metrics))],
                wspace=0.24,
            )
            color = family_colors.get(family, "#475569")
            _label_axis(
                fig.add_subplot(row_grid[0, 0]),
                family,
                family_titles.get(family, ""),
                color,
            )
            for col_idx, metric in enumerate(metrics):
                ax = fig.add_subplot(row_grid[0, col_idx + 1])
                sub = _metric_values(long_df, metric, group_col, method_order)
                if sub.empty:
                    ax.set_axis_off()
                    ax.text(
                        0.50,
                        0.50,
                        "not available",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=9.5,
                        color="#64748B",
                    )
                    continue

                sns.boxplot(
                    data=sub,
                    x=group_col,
                    y="value",
                    order=method_order,
                    hue=group_col,
                    hue_order=method_order,
                    legend=False,
                    ax=ax,
                    showfliers=False,
                    width=0.58,
                    linewidth=0.9,
                    boxprops={"alpha": 0.68},
                    palette=palette,
                )
                sns.stripplot(
                    data=sub,
                    x=group_col,
                    y="value",
                    order=method_order,
                    ax=ax,
                    size=1.7,
                    alpha=0.46,
                    color="black",
                    jitter=0.18,
                )

                panel_letter = chr(ord("A") + panel_idx)
                label = metric_labels.get(metric, metric) if metric_labels else metric
                ax.set_title(label, fontsize=10.8, fontweight="bold", pad=5)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="x", rotation=52, labelsize=7.8, pad=1.0)
                ax.tick_params(axis="y", labelsize=8.2)
                for tick in ax.get_xticklabels():
                    tick.set_horizontalalignment("right")
                    tick.set_rotation_mode("anchor")
                ax.grid(axis="y", color="#E2E8F0", linewidth=0.55, alpha=0.8)
                ax.text(
                    -0.08,
                    1.05,
                    panel_letter,
                    transform=ax.transAxes,
                    fontsize=14.5,
                    fontweight="bold",
                    color=color,
                    va="bottom",
                    ha="right",
                )

                if show_significance and reference_method is not None:
                    pairs = select_significance_pairs(
                        long_df,
                        metric=metric,
                        reference_method=reference_method,
                        group_col=group_col,
                        pair_col=pair_col,
                        top_k=2,
                    )
                    if pairs:
                        _draw_significance_brackets(
                            ax,
                            pairs,
                            method_order,
                            float(sub["value"].max()),
                        )

                axes.append(ax)
                panel_idx += 1

        if title:
            fig.text(
                0.025,
                0.965,
                title,
                ha="left",
                va="top",
                fontsize=19,
                fontweight="bold",
                color="#172033",
            )
        if subtitle:
            fig.text(
                0.025,
                0.932,
                subtitle,
                ha="left",
                va="top",
                fontsize=11.5,
                color="#475569",
            )

    return fig, axes


def _family_header(ax: plt.Axes, family: str, color: str) -> None:
    ax.add_patch(
        Rectangle(
            (0.0, 1.015),
            1.0,
            0.070,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            clip_on=False,
            zorder=8,
        )
    )
    ax.text(
        0.018,
        1.050,
        family,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="white",
        zorder=9,
    )


def _draw_missing_metric_panel(
    ax: plt.Axes,
    *,
    label: str,
    family: str,
    color: str,
) -> None:
    ax.set_facecolor("#F8FAFC")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(label, fontsize=10.6, fontweight="bold", pad=18)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.85)
    _family_header(ax, family, color)
    ax.text(
        0.50,
        0.55,
        "missing",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13.0,
        fontweight="bold",
        color=color,
    )
    ax.text(
        0.50,
        0.36,
        "not yet available\nin current result table",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.9,
        color="#64748B",
        linespacing=1.15,
    )


def create_metric_grid_figure(
    long_df: pd.DataFrame,
    *,
    metric_grid: Sequence[Sequence[str]] = METRIC_PANEL_GRID,
    group_col: str = "method",
    reference_method: str | None = None,
    method_order: Sequence[str] | None = None,
    metric_to_family: Mapping[str, str] | None = None,
    family_titles: Mapping[str, str] | None = None,
    family_colors: Mapping[str, str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    dpi: int = 300,
    palette: str = "Spectral",
    pair_col: str = "dataset_id",
    show_significance: bool = True,
    metric_labels: Mapping[str, str] | None = None,
    per_col_width: float = 2.74,
    per_row_height: float = 3.10,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Render the 24 numeric metrics as a fixed 4×6 publication grid.

    The metric families remain visible via coloured panel headers and a compact
    legend, but row/column geometry is governed by the publication layout
    contract.  Metrics with no numeric rows are still rendered as explicit
    missing panels so incomplete runs cannot silently drop expected outputs.
    """
    rows = tuple(tuple(row) for row in metric_grid)
    if not rows or any(not row for row in rows):
        raise ValueError("metric_grid must contain non-empty rows")
    nrows = len(rows)
    ncols = max(len(row) for row in rows)
    if any(len(row) != ncols for row in rows):
        raise ValueError("metric_grid rows must be rectangular")

    if method_order is None:
        method_order = long_df[group_col].dropna().unique().tolist()
    method_order = list(method_order)
    if not method_order:
        raise ValueError("method_order must be non-empty")

    metric_to_family = metric_to_family or METRIC_TO_FAMILY
    family_titles = family_titles or {}
    default_colors = {
        "BEN": "#1f5f9f",
        "DRE-UMAP": "#6C3483",
        "DRE-tSNE": "#7A4EAB",
        "LSE": "#138D75",
    }
    family_colors = {**default_colors, **(family_colors or {})}

    fig_w = 1.05 + ncols * per_col_width
    fig_h = 1.18 + nrows * per_row_height
    rcparams = {**PUBLICATION_RCPARAMS, "savefig.dpi": dpi, "figure.dpi": dpi}

    with plt.rc_context(rcparams):
        sns.set_palette(palette)
        fig, axes_grid = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_w, fig_h),
            dpi=dpi,
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.040,
            right=0.992,
            bottom=0.064,
            top=0.865 if title else 0.940,
            hspace=0.66,
            wspace=0.32,
        )

        axes: list[plt.Axes] = []
        panel_idx = 0
        used_families: list[str] = []
        for row_idx, row in enumerate(rows):
            for col_idx, metric in enumerate(row):
                ax = axes_grid[row_idx, col_idx]
                family = metric_to_family.get(metric, "metrics")
                if family not in used_families:
                    used_families.append(family)
                color = family_colors.get(family, "#475569")
                label = metric_labels.get(metric, metric) if metric_labels else metric
                sub = _metric_values(long_df, metric, group_col, method_order)

                if sub.empty:
                    _draw_missing_metric_panel(
                        ax,
                        label=label,
                        family=family,
                        color=color,
                    )
                    ax.text(
                        -0.085,
                        1.105,
                        chr(ord("A") + panel_idx),
                        transform=ax.transAxes,
                        fontsize=14.0,
                        fontweight="bold",
                        color=color,
                        va="bottom",
                        ha="right",
                    )
                    axes.append(ax)
                    panel_idx += 1
                    continue

                sns.boxplot(
                    data=sub,
                    x=group_col,
                    y="value",
                    order=method_order,
                    hue=group_col,
                    hue_order=method_order,
                    legend=False,
                    ax=ax,
                    showfliers=False,
                    width=0.58,
                    linewidth=0.9,
                    boxprops={"alpha": 0.68},
                    palette=palette,
                )
                sns.stripplot(
                    data=sub,
                    x=group_col,
                    y="value",
                    order=method_order,
                    ax=ax,
                    size=1.6,
                    alpha=0.44,
                    color="black",
                    jitter=0.18,
                )

                _family_header(ax, family, color)
                ax.set_title(label, fontsize=10.6, fontweight="bold", pad=18)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="x", rotation=52, labelsize=7.5, pad=0.8)
                ax.tick_params(axis="y", labelsize=8.1)
                for tick in ax.get_xticklabels():
                    tick.set_horizontalalignment("right")
                    tick.set_rotation_mode("anchor")
                ax.grid(axis="y", color="#E2E8F0", linewidth=0.55, alpha=0.82)
                ax.text(
                    -0.085,
                    1.105,
                    chr(ord("A") + panel_idx),
                    transform=ax.transAxes,
                    fontsize=14.0,
                    fontweight="bold",
                    color=color,
                    va="bottom",
                    ha="right",
                )

                if show_significance and reference_method is not None:
                    pairs = select_significance_pairs(
                        long_df,
                        metric=metric,
                        reference_method=reference_method,
                        group_col=group_col,
                        pair_col=pair_col,
                        top_k=2,
                    )
                    if pairs:
                        _draw_significance_brackets(
                            ax,
                            pairs,
                            method_order,
                            float(sub["value"].max()),
                        )

                axes.append(ax)
                panel_idx += 1

        if title:
            fig.text(
                0.040,
                0.972,
                title,
                ha="left",
                va="top",
                fontsize=19,
                fontweight="bold",
                color="#172033",
            )
        if subtitle:
            fig.text(
                0.040,
                0.936,
                subtitle,
                ha="left",
                va="top",
                fontsize=11.3,
                color="#475569",
            )

        legend_handles = [
            Patch(
                facecolor=family_colors.get(family, "#475569"),
                edgecolor="none",
                label=(
                    f"{family}: {family_titles[family].replace(chr(10), ' ')}"
                    if family in family_titles
                    else family
                ),
            )
            for family in used_families
        ]
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc="upper right",
                bbox_to_anchor=(0.992, 0.957),
                ncol=min(4, len(legend_handles)),
                frameon=False,
                fontsize=8.8,
                handlelength=1.15,
                columnspacing=1.05,
            )

    return fig, axes
