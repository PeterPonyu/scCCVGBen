"""Generate fig_dataset_overview.{pdf,png} from scccvgben/data/datasets.csv.

Four-panel figure:
  Panel 1 (top-left):  scRNA vs scATAC count (stacked bar, 100 each)
  Panel 2 (top-right): cell count distribution (log-scale violin per modality)
  Panel 3 (bottom-left): tissue diversity bar chart (top 15 tissues, side-by-side)
  Panel 4 (bottom-right): category breakdown (stacked horizontal bar)

Re-implemented from scratch. Does NOT import from CCVGAE_supplement/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── colour palette ────────────────────────────────────────────────────────────
C_SCRNA  = "#D35400"   # orange — consistent with prior CCVGAE figures
C_SCATAC = "#2471A3"   # blue  — consistent with prior CCVGAE figures

# ── matplotlib publication style ─────────────────────────────────────────────
plt.rcParams.update({
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
    "font.family":         "Liberation Sans",
    "savefig.bbox":        "tight",
    "savefig.dpi":         300,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.8,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.labelsize":      9,
})


def _load_datasets(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"modality", "cell_count", "tissue", "category", "drop_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"datasets.csv is missing columns: {missing}")
    df = df[df["drop_status"].str.lower() == "kept"].copy()
    df["cell_count"] = pd.to_numeric(df["cell_count"], errors="coerce")
    return df


def _panel_modality_counts(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 1: stacked bar — scRNA vs scATAC counts."""
    n_rna  = (df["modality"] == "scRNA").sum()
    n_atac = (df["modality"] == "scATAC").sum()

    ax.bar(["Benchmark\n(200 datasets)"], [n_rna],   color=C_SCRNA,  label="scRNA-seq",  width=0.4)
    ax.bar(["Benchmark\n(200 datasets)"], [n_atac],  color=C_SCATAC, label="scATAC-seq", width=0.4,
           bottom=[n_rna])

    ax.set_ylabel("Dataset count")
    ax.set_ylim(0, max(n_rna + n_atac + 20, 220))
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("A  Modality split")

    # Annotate counts
    ax.text(0, n_rna / 2, f"{n_rna}", ha="center", va="center", color="white",
            fontweight="bold", fontsize=10)
    ax.text(0, n_rna + n_atac / 2, f"{n_atac}", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)


def _panel_cell_distribution(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 2: log-scale violin of cell count per modality."""
    rna_cells  = df.loc[df["modality"] == "scRNA",  "cell_count"].dropna().values
    atac_cells = df.loc[df["modality"] == "scATAC", "cell_count"].dropna().values

    positions = [1, 2]
    data_groups = [rna_cells, atac_cells]
    colors      = [C_SCRNA, C_SCATAC]
    labels      = ["scRNA-seq", "scATAC-seq"]

    for pos, data, color, label in zip(positions, data_groups, colors, labels):
        if len(data) == 0:
            continue
        log_data = np.log10(data + 1)
        vp = ax.violinplot([log_data], positions=[pos], showmedians=True,
                           widths=0.6)
        for pc in vp["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            if part in vp:
                vp[part].set_color("black")
                vp[part].set_linewidth(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("log₁₀(cell count)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"$10^{{{int(v)}}}$" if v == int(v) else f"{10**v:.0f}"
    ))
    ax.set_title("B  Cell count distribution")


def _panel_tissue_diversity(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 3: top-15 tissue bar chart, side-by-side scRNA vs scATAC."""
    rna_tissue  = df[df["modality"] == "scRNA"]["tissue"].fillna("Unknown")
    atac_tissue = df[df["modality"] == "scATAC"]["tissue"].fillna("Unknown")

    rna_counts  = rna_tissue.value_counts()
    atac_counts = atac_tissue.value_counts()

    # Top 15 tissues by combined count
    combined = (rna_counts.add(atac_counts, fill_value=0)).sort_values(ascending=False)
    top15 = combined.head(15).index.tolist()

    rna_vals  = [rna_counts.get(t, 0) for t in top15]
    atac_vals = [atac_counts.get(t, 0) for t in top15]

    y = np.arange(len(top15))
    width = 0.35

    ax.barh(y + width / 2, rna_vals,  width, color=C_SCRNA,  label="scRNA-seq")
    ax.barh(y - width / 2, atac_vals, width, color=C_SCATAC, label="scATAC-seq")

    ax.set_yticks(y)
    ax.set_yticklabels(top15, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Dataset count")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("C  Tissue diversity (top 15)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def _panel_category_breakdown(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel 4: category breakdown — stacked horizontal bar per modality."""
    cats = df["category"].fillna("Other").unique().tolist()
    cats = sorted(set(cats))

    modalities = ["scRNA", "scATAC"]
    # Build value table
    table = {m: [((df["modality"] == m) & (df["category"].fillna("Other") == c)).sum()
                 for c in cats]
             for m in modalities}

    cmap = plt.get_cmap("tab20", len(cats))
    colors = [cmap(i) for i in range(len(cats))]

    y = np.arange(len(modalities))
    lefts = np.zeros(len(modalities))

    for i, cat in enumerate(cats):
        vals = [table[m][i] for m in modalities]
        bars = ax.barh(y, vals, left=lefts, color=colors[i], label=cat, height=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                cx = bar.get_x() + bar.get_width() / 2
                cy = bar.get_y() + bar.get_height() / 2
                ax.text(cx, cy, str(v), ha="center", va="center",
                        fontsize=6, color="white", fontweight="bold")
        lefts = lefts + np.array(vals)

    ax.set_yticks(y)
    ax.set_yticklabels(modalities)
    ax.set_xlabel("Dataset count")
    ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left",
              borderaxespad=0)
    ax.set_title("D  Category breakdown")


def make_overview_figure(csv_path: Path, out_dir: Path) -> None:
    """Main entry point — build and save the 4-panel overview figure."""
    df = _load_datasets(csv_path)
    log.info("Loaded %d kept datasets from %s", len(df), csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("scCCVGBen — 200-Dataset Benchmark Overview", fontsize=12,
                 fontweight="bold", y=1.01)

    _panel_modality_counts(axes[0, 0], df)
    _panel_cell_distribution(axes[0, 1], df)
    _panel_tissue_diversity(axes[1, 0], df)
    _panel_category_breakdown(axes[1, 1], df)

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"fig_dataset_overview.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets-csv",
        type=Path,
        default=Path(__file__).parent.parent / "scccvgben" / "data" / "datasets.csv",
        help="Path to scccvgben/data/datasets.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent.parent / "figures",
        help="Output directory for figures",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    if not args.datasets_csv.exists():
        log.error("datasets.csv not found at %s — run scripts/build_datasets_csv.py first", args.datasets_csv)
        sys.exit(1)
    make_overview_figure(args.datasets_csv, args.out_dir)
