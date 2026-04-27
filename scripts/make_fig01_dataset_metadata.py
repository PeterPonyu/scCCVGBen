"""Generate fig01_dataset_metadata.{pdf,png} from data/benchmark_manifest.csv.

Publication-style dataset metadata overview for the active 100 scRNA + 100
scATAC benchmark manifest. The layout mirrors the revised supplementary
metadata figure structure: composition, cell-count distribution, result
coverage, tissue distribution, and submission-year timeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scccvgben.figures import apply_publication_rcparams, preliminary_path  # noqa: E402

log = logging.getLogger(__name__)

C_RNA = "#D35400"
C_ATAC = "#2471A3"
C_HUMAN = "#8E44AD"
C_MOUSE = "#27AE60"
DEFAULT_TARGET = 200


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top", ha="right")


def _panel_composition(ax: plt.Axes, df: pd.DataFrame) -> None:
    modality_counts = df["modality"].value_counts().reindex(["scrna", "scatac"], fill_value=0)
    ax.pie(
        modality_counts.values,
        radius=0.72,
        colors=[C_RNA, C_ATAC],
        startangle=90,
        wedgeprops={"width": 0.30, "edgecolor": "white", "linewidth": 1.2},
    )
    species_counts = (
        df.groupby(["modality", "species_corrected"]).size()
        .reindex(pd.MultiIndex.from_product([["scrna", "scatac"], ["human", "mouse"]]), fill_value=0)
    )
    outer_sizes = [v for v in species_counts.values if v > 0]
    outer_colors = []
    for (_, species), value in species_counts.items():
        if value <= 0:
            continue
        outer_colors.append(C_HUMAN if species == "human" else C_MOUSE)
    ax.pie(
        outer_sizes,
        radius=1.08,
        colors=outer_colors,
        startangle=90,
        wedgeprops={"width": 0.30, "edgecolor": "white", "linewidth": 0.8},
    )
    ax.text(0, 0.04, f"{len(df)}", ha="center", va="center", fontsize=19, color="#1C2833")
    ax.text(0, -0.20, "datasets", ha="center", va="center", fontsize=10, color="#555")
    ax.set_title("Dataset composition", loc="left", fontsize=13, pad=8)
    handles = [
        mpatches.Patch(color=C_RNA, label=f"scRNA (n={modality_counts['scrna']})"),
        mpatches.Patch(color=C_ATAC, label=f"scATAC (n={modality_counts['scatac']})"),
        mpatches.Patch(color=C_HUMAN, label="human"),
        mpatches.Patch(color=C_MOUSE, label="mouse"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, frameon=False, fontsize=9)
    _panel_label(ax, "A")


def _panel_cell_violin(ax: plt.Axes, df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["cell_count"]).copy()
    plot_df["cell_count_log10"] = np.log10(plot_df["cell_count"].clip(lower=1))
    sns.violinplot(
        data=plot_df,
        x="modality",
        y="cell_count_log10",
        order=["scrna", "scatac"],
        hue="modality",
        palette={"scrna": C_RNA, "scatac": C_ATAC},
        legend=False,
        inner="quartile",
        linewidth=0.8,
        cut=0,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="modality",
        y="cell_count_log10",
        order=["scrna", "scatac"],
        color="black",
        size=2.4,
        alpha=0.45,
        jitter=0.18,
        ax=ax,
    )
    for idx, modality in enumerate(["scrna", "scatac"]):
        sub = plot_df[plot_df["modality"] == modality]
        if sub.empty:
            continue
        med = int(sub["cell_count"].median())
        total = sub["cell_count"].sum() / 1e6
        ax.text(idx, sub["cell_count_log10"].quantile(0.88),
                f"Med {med:,}\nTotal {total:.1f}M", ha="center", va="bottom",
                fontsize=9, color=C_RNA if modality == "scrna" else C_ATAC,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.85,
                      "edgecolor": C_RNA if modality == "scrna" else C_ATAC, "linewidth": 0.6})
    ax.set_xlabel("")
    ax.set_ylabel("Cell count (log10)")
    ax.set_title("Cell-count distribution", loc="left", fontsize=13, pad=8)
    _panel_label(ax, "B")


def _panel_coverage(ax: plt.Axes, df: pd.DataFrame, results_root: Path) -> None:
    rows = []
    for modality, subdir in [("scrna", "reconciled/scrna"), ("scatac", "reconciled/scatac")]:
        target = int((df["modality"] == modality).sum())
        observed = len(list((results_root / subdir).glob("*.csv"))) if (results_root / subdir).is_dir() else 0
        rows.append({"modality": modality, "target": target, "observed": min(observed, target)})
    cov = pd.DataFrame(rows)
    ax.bar(cov["modality"], cov["target"], color="#E5E7EB", label="manifest target")
    ax.bar(cov["modality"], cov["observed"], color=[C_RNA, C_ATAC], label="available result tables")
    for idx, row in cov.iterrows():
        ax.text(idx, row["observed"] + 1, f"{row['observed']}/{row['target']}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(cov["target"].max(), cov["observed"].max()) * 1.18)
    ax.set_ylabel("Datasets")
    ax.set_title("Current result-table coverage", loc="left", fontsize=13, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _panel_label(ax, "C")


def _panel_tissue(ax: plt.Axes, df: pd.DataFrame, top_k: int = 12) -> None:
    """Tissue distribution panel.

    The scATAC half of the manifest still has ~73/100 entries lacking a
    canonical tissue label (CCVGAE never annotated those samples). We
    exclude the catch-all ``other`` / ``unknown`` buckets from the top-K
    ranking and stamp the un-annotated count as a footnote so the panel
    is not silently skewed by the annotation gap.
    """
    BLOCKLIST = {"other", "unknown", "mixed", ""}
    raw = df.assign(tissue=df["tissue"].fillna("other").str.lower())
    n_unannotated = int(raw["tissue"].isin(BLOCKLIST).sum())
    raw = raw[~raw["tissue"].isin(BLOCKLIST)]

    tissue_counts = raw.groupby(["tissue", "modality"]).size().unstack(fill_value=0)
    tissue_counts["total"] = tissue_counts.sum(axis=1)
    tissue_counts = tissue_counts.sort_values("total", ascending=True).tail(top_k)
    y = np.arange(len(tissue_counts))
    left = np.zeros(len(tissue_counts))
    for modality, color in [("scrna", C_RNA), ("scatac", C_ATAC)]:
        vals = tissue_counts.get(modality, pd.Series(0, index=tissue_counts.index)).to_numpy()
        ax.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.4, label=modality)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(tissue_counts.index)
    ax.set_xlabel("Datasets")
    ax.set_ylabel("")
    ax.set_title(f"Top-{top_k} annotated tissue sources",
                 loc="left", fontsize=13, pad=8)
    # No per-panel legend: panel A already shows the global scRNA/scATAC
    # modality legend, so a duplicate here would just collide with the
    # 'tumor' bar that fills the upper-right.
    if n_unannotated:
        ax.text(
            0.99, 0.02,
            f"{n_unannotated} datasets without canonical tissue label",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#475569", style="italic",
        )
    _panel_label(ax, "D")


def _panel_year(ax: plt.Axes, df: pd.DataFrame) -> None:
    years = pd.to_datetime(df["geo_submission_date"], errors="coerce").dt.year
    plot_df = df.assign(year=years).dropna(subset=["year"])
    by_year = plot_df.groupby([plot_df["year"].astype(int), "modality"]).size().unstack(fill_value=0)
    by_year = by_year.sort_index()
    for modality, color in [("scrna", C_RNA), ("scatac", C_ATAC)]:
        if modality in by_year:
            ax.plot(by_year.index, by_year[modality], marker="o", lw=1.8,
                    color=color, label=modality)
            ax.fill_between(by_year.index, 0, by_year[modality], alpha=0.16, color=color)
    ax.set_xlabel("GEO submission year")
    ax.set_ylabel("Datasets")
    ax.set_title("Submission-year timeline", loc="left", fontsize=13, pad=8)
    ax.legend(frameon=False, fontsize=8)
    _panel_label(ax, "E")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "data" / "benchmark_manifest.csv")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--partial-ok", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_publication_rcparams()

    if not args.manifest.exists():
        log.error("manifest missing: %s", args.manifest)
        return 1
    df = pd.read_csv(args.manifest)
    df["cell_count"] = pd.to_numeric(df["cell_count"], errors="coerce")
    n_datasets = len(df)
    log.info("manifest rows: %d (target %d)", n_datasets, args.target_n)

    if n_datasets < args.target_n and not args.partial_ok:
        log.error("only %d/%d datasets — pass --partial-ok", n_datasets, args.target_n)
        return 1

    fig = plt.figure(figsize=(13.5, 8.4), dpi=300)
    gs_top = gridspec.GridSpec(1, 3, figure=fig, left=0.04, right=0.98,
                               top=0.90, bottom=0.54, wspace=0.36,
                               width_ratios=[1.15, 1.0, 0.95])
    gs_bot = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.98,
                               top=0.42, bottom=0.08, wspace=0.34)

    _panel_composition(fig.add_subplot(gs_top[0, 0]), df)
    _panel_cell_violin(fig.add_subplot(gs_top[0, 1]), df)
    _panel_coverage(fig.add_subplot(gs_top[0, 2]), df, args.results_root)
    _panel_tissue(fig.add_subplot(gs_bot[0, 0]), df)
    _panel_year(fig.add_subplot(gs_bot[0, 1]), df)

    suffix = "" if n_datasets >= args.target_n else f" (PRELIMINARY: {n_datasets}/{args.target_n})"
    fig.suptitle(f"Fig 01 — active benchmark metadata{suffix}", fontsize=15, y=0.985)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig01_dataset_metadata"
    pdf_path = args.out_dir / preliminary_path(stem, n_datasets, args.target_n,
                                               suffix=".pdf").name
    png_path = args.out_dir / preliminary_path(stem, n_datasets, args.target_n,
                                               suffix=".png").name
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    log.info("wrote %s", pdf_path)
    log.info("wrote %s", png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
