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

from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from ._significance import select_significance_pairs

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
