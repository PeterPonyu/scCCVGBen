"""DataFrame-native publication figure helper.

Visual contract reproduced from the read-only reference implementation
WITHOUT importing its analyzer module or coupling to its state machine.
All figures consume a pre-melted long DataFrame.

Visual contract:
- box + strip overlay (seaborn).
- Wilcoxon Holm-corrected significance brackets, or dense marker rows when a
  panel would otherwise need many stacked brackets.
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
from .fonts import register_arial_with_matplotlib
from .metrics import METRIC_PANEL_GRID, METRIC_TO_FAMILY

register_arial_with_matplotlib()

PaletteSpec = str | Sequence[str] | Mapping[str, str]

METHOD_COLORS: dict[str, str] = {
    "VAE": "#D99B1E",
    "Base": "#D99B1E",
    "Base\nVAE": "#D99B1E",
    "CenVAE": "#2A9D8F",
    "Centroid": "#2A9D8F",
    "Centroid\nvar.": "#2A9D8F",
    "CouVAE": "#E76F51",
    "Coupling": "#E76F51",
    "Coupling\nvar.": "#E76F51",
    "GAT-VAE": "#6D5DD3",
    "GAT\nvar.": "#6D5DD3",
    "CCVGAE": "#111827",
    "scCCVGBen": "#111827",
    "GAT": "#6D5DD3",
    "GATv2": "#7C3AED",
    "Transformer": "#2563EB",
    "SuperGAT": "#0891B2",
    "GCN": "#16A34A",
    "SAGE": "#84CC16",
    "GIN": "#F59E0B",
    "Cheb": "#EA580C",
    "EdgeConv": "#DC2626",
    "ARMA": "#DB2777",
    "SG": "#9333EA",
    "TAG": "#475569",
    "Graph": "#64748B",
    "SSG": "#0F766E",
    "PCA": "#1D4ED8",
    "KPCA": "#0E7490",
    "ICA": "#059669",
    "FA": "#65A30D",
    "NMF": "#CA8A04",
    "TSVD": "#C2410C",
    "DICL": "#BE123C",
    "scVI": "#7C2D12",
    "DIP": "#7E22CE",
    "INFO": "#C026D3",
    "TC": "#4F46E5",
    "highBeta": "#0369A1",
    "LSI": "#2563EB",
    "PeakVI": "#059669",
    "PoissonVI": "#CA8A04",
    "cisTopic": "#BE123C",
    "kNN-euc": "#111827",
    "kNN-cos": "#2563EB",
    "SNN": "#0F766E",
    "Mutual-kNN": "#D97706",
    "Gaussian": "#C2410C",
}

_FALLBACK_METHOD_COLORS: tuple[str, ...] = (
    "#2563EB",
    "#059669",
    "#D97706",
    "#DC2626",
    "#7C3AED",
    "#0891B2",
    "#BE123C",
    "#4B5563",
)


def palette_for_methods(methods: Sequence[str]) -> dict[str, str]:
    """Stable method -> colour mapping for multi-method manuscript grids."""
    palette: dict[str, str] = {}
    fallback_i = 0
    for method in methods:
        if method in palette:
            continue
        color = METHOD_COLORS.get(str(method))
        if color is None:
            color = _FALLBACK_METHOD_COLORS[fallback_i % len(_FALLBACK_METHOD_COLORS)]
            fallback_i += 1
        palette[str(method)] = color
    return palette


PUBLICATION_RCPARAMS: dict = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "Arial",
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
    register_arial_with_matplotlib()
    mpl.rcParams.update(PUBLICATION_RCPARAMS)
    sns.set_palette("Spectral")
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def _apply_plot_palette(palette: PaletteSpec) -> None:
    if isinstance(palette, Mapping):
        sns.set_palette(list(palette.values()))
    else:
        sns.set_palette(palette)


def _hundreds_tick_label(value: float) -> str:
    scaled = value / 100.0
    if scaled == 0:
        return "0"
    if abs(scaled) >= 1000 or abs(scaled) < 0.01:
        return f"{scaled:.1e}"
    return f"{scaled:g}"


def _draw_significance_brackets(
    ax: plt.Axes,
    pairs: list[tuple[str, str, float]],
    method_order: Sequence[str],
    y_data_max: float,
    *,
    marker_fontsize: float = 13.5,
    ns_fontsize: float = 11.0,
    step_fraction: float = 0.165,
    text_pad_fraction: float = 0.20,
    dense_marker_threshold: int | None = None,
) -> None:
    if not pairs:
        return
    positions = {m: i for i, m in enumerate(method_order)}
    # Order brackets by horizontal span ascending so the narrowest sits closest
    # to the data and the widest stacks on top. All brackets share the
    # reference-method left leg, so this stacking order keeps the visible legs
    # cleanly nested instead of crossing each other.
    drawable: list[tuple[int, int, float]] = []
    for a, b, p in pairs:
        if a not in positions or b not in positions:
            continue
        x1, x2 = positions[a], positions[b]
        drawable.append((x1, x2, p))
    if not drawable:
        return
    drawable.sort(key=lambda t: abs(t[1] - t[0]))

    span = max(y_data_max, 1e-9)
    # Slightly more vertical room between stacked brackets so the marker that
    # sits above bracket j cannot graze the leg of bracket j+1.
    step = span * step_fraction
    base = y_data_max + step
    leg = step * 0.20
    # Marker baseline tucked close to the bracket cap (small offset, smaller
    # font) so the asterisks read as belonging to *this* bracket instead of
    # floating in the space beneath the next bracket.
    text_pad = step * text_pad_fraction

    def marker_for(p: float) -> tuple[str, float, str]:
        if p < 0.001:
            return "***", marker_fontsize, "bold"
        if p < 0.01:
            return "**", marker_fontsize, "bold"
        if p < 0.05:
            return "*", marker_fontsize, "bold"
        return "ns", ns_fontsize, "normal"

    if dense_marker_threshold is not None and len(drawable) > dense_marker_threshold:
        y = y_data_max + step * 0.72
        for _x1, x2, p in sorted(drawable, key=lambda t: t[1]):
            sig, fs, fw = marker_for(p)
            ax.text(
                x2,
                y,
                sig,
                ha="center",
                va="bottom",
                fontsize=fs,
                fontweight=fw,
                color="#1f2937",
            )
        ax.set_ylim(top=y_data_max + step * 2.2)
        return

    for j, (x1, x2, p) in enumerate(drawable):
        y = base + j * step
        ax.plot([x1, x1, x2, x2], [y - leg, y, y, y - leg],
                lw=1.4, color="#1f2937", solid_capstyle="round")
        sig, fs, fw = marker_for(p)
        ax.text((x1 + x2) / 2, y + text_pad, sig,
                ha="center", va="bottom", fontsize=fs,
                fontweight=fw, color="#1f2937")
    ax.set_ylim(top=base + (len(drawable) + 1.4) * step)


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
    palette: PaletteSpec = "Spectral",
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
        _apply_plot_palette(palette)
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
                cap = max(1, len(method_order) - 1)
                pairs = select_significance_pairs(
                    long_df, metric=metric,
                    reference_method=reference_method,
                    group_col=group_col, pair_col=pair_col,
                    top_k=cap,
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
    palette: PaletteSpec = "Spectral",
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
        _apply_plot_palette(palette)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        outer_grid = fig.add_gridspec(
            nrows,
            1,
            left=0.025,
            right=0.992,
            bottom=0.055,
            top=0.90 if title else 0.96,
            hspace=0.68,
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
                family_xtick_rotation = 52
                ax.tick_params(axis="x", rotation=family_xtick_rotation, labelsize=9.4, pad=3.0)
                ax.tick_params(axis="y", labelsize=8.6)
                tick_ha = "center" if abs(family_xtick_rotation) < 1e-6 else "right"
                tick_mode = "default" if abs(family_xtick_rotation) < 1e-6 else "anchor"
                for tick in ax.get_xticklabels():
                    tick.set_horizontalalignment(tick_ha)
                    tick.set_rotation_mode(tick_mode)
                ax.grid(axis="y", color="#E2E8F0", linewidth=0.55, alpha=0.8)
                ax.text(
                    -0.08,
                    1.05,
                    panel_letter,
                    transform=ax.transAxes,
                    fontsize=16,
                    fontweight="bold",
                    color="black",
                    va="bottom",
                    ha="right",
                )

                if show_significance and reference_method is not None:
                    cap = max(1, len(method_order) - 1)
                    pairs = select_significance_pairs(
                        long_df,
                        metric=metric,
                        reference_method=reference_method,
                        group_col=group_col,
                        pair_col=pair_col,
                        top_k=cap,
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


def _compact_family_label(family: str) -> str:
    """Short label for narrow metric-grid header bars."""
    if family.startswith("DRE-"):
        return "DRE"
    return family


def _family_header(ax: plt.Axes, family: str, color: str,
                   metric_label: str | None = None) -> None:
    ax.add_patch(
        Rectangle(
            (0.0, 1.015),
            1.0,
            0.114,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            clip_on=False,
            zorder=8,
        )
    )
    family_label = _compact_family_label(family)
    metric_text = str(metric_label or "")
    long_metric_label = len(metric_text.replace("$", "").replace("\\", "")) >= 18
    family_fontsize = 10.7 if long_metric_label else 11.8
    metric_fontsize = 11.0 if long_metric_label else 12.8
    ax.text(
        0.022,
        1.072,
        family_label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=family_fontsize,
        fontweight="bold",
        color="white",
        zorder=9,
    )
    if metric_label:
        # Put the per-panel metric name on the right of the family bar — the
        # bar already carries the metric category, so the panel title would
        # otherwise duplicate that information.
        ax.text(
            0.978,
            1.072,
            metric_label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=metric_fontsize,
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
    ax.set_title(label, fontsize=12.6, fontweight="bold", pad=20)
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
        fontsize=16.0,
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
        fontsize=11.6,
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
    palette: PaletteSpec = "Spectral",
    pair_col: str = "dataset_id",
    show_significance: bool = True,
    metric_labels: Mapping[str, str] | None = None,
    per_col_width: float = 3.12,
    per_row_height: float = 3.30,
    panel_labels_enabled: bool = True,
    panel_label_letter: str | None = None,
    significance_pairs_per_panel: int | None = None,
    xtick_rotation: float = 52,
    xtick_labelsize: float = 13.5,
    ytick_labelsize: float = 13.5,
    hspace: float = 0.78,
    box_width: float = 0.62,
    strip_size: float = 2.35,
    strip_jitter: float = 0.18,
    strip_alpha: float = 0.54,
    significance_marker_fontsize: float = 10.0,
    significance_ns_fontsize: float = 8.5,
    significance_step_fraction: float = 0.135,
    significance_text_pad_fraction: float = 0.08,
    significance_dense_marker_threshold: int | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Render the curated 20 numeric metrics as a fixed 4×5 publication grid.

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
        _apply_plot_palette(palette)
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
            bottom=0.072,
            top=0.866 if title else 0.952,
            hspace=hspace,
            wspace=0.32,
        )

        axes: list[plt.Axes] = []
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
                    axes.append(ax)
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
                    width=box_width,
                    linewidth=1.12,
                    boxprops={"alpha": 0.70},
                    palette=palette,
                )
                sns.stripplot(
                    data=sub,
                    x=group_col,
                    y="value",
                    order=method_order,
                    ax=ax,
                    size=strip_size,
                    alpha=strip_alpha,
                    color="black",
                    jitter=strip_jitter,
                )

                # Family bar carries BOTH the family name (left) and the
                # specific metric name (right) — so we drop the separate
                # `ax.set_title` to avoid duplicate information and recover
                # the vertical space it used to consume.
                _family_header(ax, family, color, metric_label=label)
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="x", rotation=xtick_rotation, labelsize=xtick_labelsize, pad=1.4)
                ax.tick_params(axis="y", labelsize=ytick_labelsize)
                tick_ha = "center" if abs(xtick_rotation) < 1e-6 else "right"
                tick_mode = "default" if abs(xtick_rotation) < 1e-6 else "anchor"
                for tick in ax.get_xticklabels():
                    tick.set_horizontalalignment(tick_ha)
                    tick.set_rotation_mode(tick_mode)
                ax.grid(axis="y", color="#E2E8F0", linewidth=0.55, alpha=0.82)
                # Auto sci-style y-axis when the metric range crosses 1e3
                # (or sits below 1e-3) — keeps tick text from overflowing
                # the panel and unifies typography across very different
                # numeric scales. We force a fresh ScalarFormatter because
                # seaborn boxplot may have set a non-standard formatter and
                # `ax.ticklabel_format` then becomes a silent no-op.
                y_abs_max = float(sub["value"].abs().max())
                force_hundreds_scale = metric == "CAL" or metric.startswith("K_max")
                if force_hundreds_scale:
                    from matplotlib.ticker import FuncFormatter as _FF
                    formatter = _FF(lambda value, _pos: _hundreds_tick_label(value))
                    ax.yaxis.set_major_formatter(formatter)
                    ax.text(
                        0.025,
                        0.955,
                        r"$\times 10^{2}$",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9.8,
                        color="#475569",
                        zorder=9.5,
                    )
                elif y_abs_max >= 1000.0 or (0 < y_abs_max < 1e-3):
                    from matplotlib.ticker import ScalarFormatter as _SF
                    fmt = _SF(useMathText=True)
                    fmt.set_scientific(True)
                    fmt.set_powerlimits((-3, 3))
                    ax.yaxis.set_major_formatter(fmt)
                    ax.yaxis.get_offset_text().set_fontsize(10.5)

                if show_significance and reference_method is not None:
                    # Match CCVGAE's policy: show ALL pairwise comparisons
                    # against the reference method (one bracket per non-reference
                    # method), not just the top-k.
                    cap = (
                        significance_pairs_per_panel
                        if significance_pairs_per_panel is not None
                        else max(1, len(method_order) - 1)
                    )
                    pairs = select_significance_pairs(
                        long_df,
                        metric=metric,
                        reference_method=reference_method,
                        group_col=group_col,
                        pair_col=pair_col,
                        top_k=cap,
                    )
                    if pairs:
                        _draw_significance_brackets(
                            ax,
                            pairs,
                            method_order,
                            float(sub["value"].max()),
                            marker_fontsize=significance_marker_fontsize,
                            ns_fontsize=significance_ns_fontsize,
                            step_fraction=significance_step_fraction,
                            text_pad_fraction=significance_text_pad_fraction,
                            dense_marker_threshold=significance_dense_marker_threshold,
                        )

                axes.append(ax)

        if panel_labels_enabled:
            for row_idx, row in enumerate(rows):
                if panel_label_letter is not None and row_idx > 0:
                    continue
                letter = (
                    panel_label_letter
                    if panel_label_letter is not None
                    else chr(ord("A") + row_idx)
                )
                axes_grid[row_idx, 0].text(
                    -0.165,
                    1.085,
                    letter,
                    transform=axes_grid[row_idx, 0].transAxes,
                    fontsize=38,
                    fontweight="bold",
                    color="black",
                    va="center",
                    ha="center",
                    zorder=10,
                )

        if title:
            fig.text(
                0.040,
                0.972,
                title,
                ha="left",
                va="top",
                fontsize=21.5,
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
                fontsize=12.4,
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
        if legend_handles and (title or subtitle):
            fig.legend(
                handles=legend_handles,
                loc="upper right",
                bbox_to_anchor=(0.992, 0.957),
                ncol=min(4, len(legend_handles)),
                frameon=False,
                fontsize=10.2,
                handlelength=1.15,
                columnspacing=1.05,
            )

    return fig, axes
