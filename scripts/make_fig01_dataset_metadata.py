"""Publication-grade metadata overview of the active scCCVGBen benchmark.

Three panels: dataset/species composition, cell-count distribution, and
submission-year timeline. Used as the upper component of Figure 2 (composite)
and as a standalone reference plate.

Outputs:
- figures/fig01_dataset_metadata.{pdf,png}  (single canonical stem)
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


def _format_total(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return f"{n:,}"

C_RNA = "#D35400"
C_ATAC = "#2471A3"
C_HUMAN = "#8E44AD"
C_MOUSE = "#27AE60"
C_OTHER_SPECIES = "#F39C12"
C_UNASSIGNED = "#7F8C8D"
DEFAULT_TARGET = 200


def _panel_label(ax: plt.Axes, label: str, *, x: float = -0.10) -> None:
    ax.text(x, 1.08, label, transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="top", ha="left")


def _panel_composition(ax: plt.Axes, df: pd.DataFrame) -> None:
    modality_counts = df["modality"].value_counts().reindex(["scrna", "scatac"], fill_value=0)
    # Pie sits in the LEFT half of its subplot so that the species/modality
    # legend can be placed vertically along the right side without overlap.
    ax.set_xlim(-1.9, 3.1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    pie_center = (-0.55, 0.0)
    ax.pie(
        modality_counts.values,
        radius=0.70,
        center=pie_center,
        colors=[C_RNA, C_ATAC],
        startangle=90,
        wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 1.2},
    )
    # Outer ring shows the SPECIES split aggregated across modalities — one
    # purple wedge for human, one green wedge for mouse. Splitting by
    # (modality, species) made the same colour appear twice and obscured
    # the simple human/mouse story.
    species = (
        df["species_corrected"]
        .fillna("unassigned")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": "unassigned", "nan": "unassigned", "none": "unassigned"})
    )
    species_counts = species.value_counts()
    human_total = int(species_counts.get("human", 0))
    mouse_total = int(species_counts.get("mouse", 0))
    unassigned_total = int(species_counts.get("unassigned", 0))
    other_species_total = int(len(df) - human_total - mouse_total - unassigned_total)
    species_sizes = [human_total, mouse_total, other_species_total]
    species_colors = [C_HUMAN, C_MOUSE, C_OTHER_SPECIES]
    if unassigned_total:
        species_sizes.append(unassigned_total)
        species_colors.append(C_UNASSIGNED)
    ax.pie(
        species_sizes,
        radius=1.22,
        center=pie_center,
        colors=species_colors,
        startangle=90,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.0},
    )
    ax.text(pie_center[0], pie_center[1] + 0.06, f"{len(df)}", ha="center", va="center",
            fontsize=26, fontweight="bold", color="#1C2833")
    ax.text(pie_center[0], pie_center[1] - 0.22, "datasets", ha="center", va="center",
            fontsize=13, color="#1F2D3D")
    ax.set_title("Dataset composition", loc="left", fontsize=16, pad=4)
    handles = [
        mpatches.Patch(color=C_RNA, label=f"scRNA ({modality_counts['scrna']})"),
        mpatches.Patch(color=C_ATAC, label=f"scATAC ({modality_counts['scatac']})"),
        mpatches.Patch(color=C_HUMAN, label=f"human ({human_total})"),
        mpatches.Patch(color=C_MOUSE, label=f"mouse ({mouse_total})"),
        mpatches.Patch(color=C_OTHER_SPECIES, label=f"other spp. ({other_species_total})"),
    ]
    if unassigned_total:
        handles.append(mpatches.Patch(color=C_UNASSIGNED, label=f"pending metadata (n={unassigned_total})"))
    # Legend stacked vertically along the right side of the pie keeps the
    # category list compact and avoids the two-row footer that previously
    # collided with the bottom architecture half.
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.86, 0.0),
              bbox_transform=ax.transData, ncol=1, frameon=False,
              fontsize=9.2, handlelength=1.15, borderaxespad=0.0,
              labelspacing=0.28)
    # Panel-label "A" pinned to the axes left edge so it lines up with
    # panel D's left margin in the architecture composite below.
    _panel_label(ax, "A", x=-0.32)


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
    y_top = float(plot_df["cell_count_log10"].max()) + 0.45
    ax.set_ylim(top=y_top)
    for idx, modality in enumerate(["scrna", "scatac"]):
        sub = plot_df[plot_df["modality"] == modality]
        if sub.empty:
            continue
        med = int(sub["cell_count"].median())
        total_int = int(sub["cell_count"].sum())
        ax.text(idx, y_top - 0.12,
                f"Med {med:,}\nTotal {_format_total(total_int)}", ha="center", va="top",
                fontsize=12, color=C_RNA if modality == "scrna" else C_ATAC,
                bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "alpha": 0.95,
                      "edgecolor": C_RNA if modality == "scrna" else C_ATAC,
                      "linewidth": 0.9}, zorder=5)
    ax.set_xlabel("")
    ax.set_ylabel("Cell count (log10)", fontsize=13)
    ax.set_title("Cell-count distribution", loc="left", fontsize=16, pad=4)
    ax.tick_params(axis="both", labelsize=12)
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
    ymax = max(cov["target"].max(), cov["observed"].max()) * 1.18
    ax.set_ylim(0, ymax)
    for idx, row in cov.iterrows():
        ax.text(idx, row["observed"] + ymax * 0.015,
                f"{row['observed']}/{row['target']}",
                ha="center", va="bottom", fontsize=14, fontweight="bold", zorder=5)
    ax.set_ylabel("Datasets", fontsize=14)
    ax.set_title("Current result-table coverage", loc="left", fontsize=18, pad=10)
    # Legend in the upper-left corner keeps clear space below for the bar-value
    # annotations now that ymax padding is tighter.
    ax.legend(frameon=False, fontsize=12, loc="upper left",
              bbox_to_anchor=(0.02, 0.99))
    ax.tick_params(axis="both", labelsize=13)
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
        ax.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.8, label=modality)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(tissue_counts.index, fontsize=13)
    ax.set_xlabel("Datasets", fontsize=14)
    ax.set_ylabel("")
    ax.set_title(f"Top-{top_k} annotated tissue sources",
                 loc="left", fontsize=18, pad=10)
    ax.tick_params(axis="x", labelsize=13)
    # No per-panel legend: panel A already shows the global scRNA/scATAC
    # modality legend, so a duplicate here would just collide with the
    # 'tumor' bar that fills the upper-right.
    if n_unannotated:
        ax.text(
            0.5, -0.18,
            f"{n_unannotated} datasets without canonical tissue label",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, color="#334155",
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
    ax.set_xlabel("GEO submission year", fontsize=13)
    ax.set_ylabel("Datasets", fontsize=13)
    ax.set_title("Submission-year timeline", loc="left", fontsize=16, pad=4)
    ax.legend(frameon=False, fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    _panel_label(ax, "C")


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

    # Render at the same horizontal extent as fig02 (20.2 in) so the stitched
    # Figure 2 composite shows equivalent text scale across both halves.
    # Single row keeps the metadata strip compact above the architecture half.
    fig = plt.figure(figsize=(20.2, 4.0), dpi=300)
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.012, right=0.985,
                           top=0.94, bottom=0.14, wspace=0.36,
                           width_ratios=[1.50, 1.0, 1.0])

    _panel_composition(fig.add_subplot(gs[0, 0]), df)
    _panel_cell_violin(fig.add_subplot(gs[0, 1]), df)
    _panel_year(fig.add_subplot(gs[0, 2]), df)

    # Suptitle moved to the LaTeX figure caption per paper-layout policy.

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    stem = "fig01_dataset_metadata"
    pdf_path = args.out_dir / preliminary_path(stem, n_datasets, args.target_n,
                                               suffix=".pdf").name
    png_path = args.out_dir / preliminary_path(stem, n_datasets, args.target_n,
                                               suffix=".png").name
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    written.extend([pdf_path, png_path])
    for path in written:
        log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
