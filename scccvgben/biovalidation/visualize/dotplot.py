"""GO BP enrichment dotplot per latent dimension.

Inspired by ``sc.pl.dotplot`` and the legacy CCVGAE notebooks: each row is a
GO term, each column a latent dim, dot size encodes the overlap percentage
(``n_overlap / n_term``), dot colour encodes ``-log10(padj)``.

The panel is laid out so that:
  * up to ``max_terms_total`` distinct GO terms appear on the y-axis
  * x-axis spans every latent dim that returned any term
  * an inset colorbar in the upper-right corner shows the colour scale
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .panel import render_placeholder


def render_gobp_dotplot(
    ax: plt.Axes,
    enrich_df: pd.DataFrame,
    *,
    title: str = "H · GO BP enrichment per latent dim (top correlated genes)",
    max_terms_total: int = 20,
    cmap: str = "viridis",
    size_scale: float = 480.0,
) -> None:
    """Render a dotplot of GO BP terms × latent dims.

    Parameters
    ----------
    ax : pre-allocated host axis (composer slot)
    enrich_df : DataFrame from :func:`go_bp_enrichment_per_dim`
    max_terms_total : keep the top-N most enriched terms across dims
    cmap : matplotlib colormap for ``neg_log10_padj``
    size_scale : multiplied by ``percent`` to get scatter size
    """
    if enrich_df is None or enrich_df.empty:
        render_placeholder(ax, "no GO BP enrichment (gseapy missing or no genes)")
        return

    # Aggregate: keep the most-enriched (lowest padj) row per (term, dim) pair
    agg = (
        enrich_df.sort_values("padj")
                 .drop_duplicates(["term", "dim"], keep="first")
                 .copy()
    )
    # Pick top terms by best-padj across all dims
    best_per_term = agg.groupby("term")["neg_log10_padj"].max().sort_values(ascending=False)
    top_terms = best_per_term.head(max_terms_total).index.tolist()
    agg = agg[agg["term"].isin(top_terms)]
    if agg.empty:
        render_placeholder(ax, "no terms above threshold")
        return

    # Order: dims ascending, terms by best-padj descending
    dims = sorted(agg["dim"].unique())
    terms = top_terms  # already best-first

    term_idx = {t: i for i, t in enumerate(terms)}
    dim_idx  = {d: i for i, d in enumerate(dims)}

    xs = agg["dim"].map(dim_idx).to_numpy()
    ys = agg["term"].map(term_idx).to_numpy()
    sizes = (agg["percent"].to_numpy() * size_scale) + 4
    colors = agg["neg_log10_padj"].to_numpy()

    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap=cmap, alpha=0.92,
                    edgecolors="white", linewidths=0.4)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"d{d}" for d in dims], fontsize=9)
    ax.set_yticks(range(len(terms)))
    # Truncate term names. With the case figure left margin now 0.18
    # (≈2.9 in label space at 16-in width) and 9-pt sans, ~60 chars fit
    # comfortably without colliding into the plot area.
    ax.set_yticklabels([_pretty_term(t)[:60] for t in terms], fontsize=9)
    ax.set_xlim(-0.5, len(dims) - 0.5)
    ax.set_ylim(-0.5, len(terms) - 0.5)
    ax.invert_yaxis()  # best-padj at top
    ax.set_xlabel("latent dim", fontsize=9)
    ax.set_title(title, pad=6, fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, color="#E2E8F0", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#CBD5E1")
        s.set_linewidth(0.6)

    # Inline colorbar (small, top-right) and size legend
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.018, shrink=0.55, pad=0.006)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("-log10(padj)", fontsize=8)

    # Size legend overlay (lower-right corner)
    sample_pcts = [0.05, 0.15, 0.30]
    handles = [
        ax.scatter([], [], s=p * size_scale + 4, color="#94A3B8",
                   edgecolors="white", linewidths=0.3, label=f"{int(p*100)}%")
        for p in sample_pcts
    ]
    ax.legend(handles=handles, loc="lower right", title="overlap",
              fontsize=8, title_fontsize=8, frameon=False, handlelength=1.2,
              labelspacing=0.4, borderpad=0.2)


def _pretty_term(t: str) -> str:
    """Strip the MSigDB ``GOBP_`` prefix and lowercase."""
    raw = t
    for pf in ("GOBP_", "GO_BP_", "REACTOME_", "KEGG_"):
        if raw.startswith(pf):
            raw = raw[len(pf):]
            break
    return raw.replace("_", " ").lower()
