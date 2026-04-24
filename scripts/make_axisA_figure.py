"""Generate fig_axisA_encoder_ranking.{pdf,png} from results/encoder_sweep/*.csv.

Two panels:
  Panel 1: encoder mean-rank heatmap (12 encoders × 4 primary metrics),
           attention-family rows highlighted with a coloured band.
  Panel 2: bar chart of attention vs message-passing family mean rank
           with Wilcoxon p-value overlay (loaded from results/stats/axisA_attention_vs_mp.csv).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRIMARY_METRICS = ["ARI", "NMI", "ASW", "distance_correlation_umap"]
C_ATTENTION     = "#E8C4A4"   # warm sand for attention-family band
C_MP            = "#C4D4E8"   # cool blue for MP-family band
C_HEAT_LOW      = "#FFFFFF"
C_HEAT_HIGH     = "#D35400"

plt.rcParams.update({
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
    "font.family":     "Liberation Sans",
    "savefig.bbox":    "tight",
    "savefig.dpi":     300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         9,
    "axes.titlesize":    10,
})

ATTENTION_ENCODERS = {"GAT", "GATv2", "TransformerConv", "SuperGAT"}


def _load_encoder_results(sweep_dir: Path) -> pd.DataFrame:
    """Load all encoder_sweep CSVs and return a long DataFrame."""
    frames = []
    for csv_f in sweep_dir.glob("*.csv"):
        df = pd.read_csv(csv_f)
        df["dataset_key"] = csv_f.stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSVs found in {sweep_dir}")
    combined = pd.concat(frames, ignore_index=True)
    # Normalise encoder column name
    enc_col = next((c for c in combined.columns if c.lower() in ("encoder", "method")), None)
    if enc_col and enc_col != "encoder":
        combined = combined.rename(columns={enc_col: "encoder"})
    return combined


def _compute_mean_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """For each metric, rank encoders per dataset, then average across datasets."""
    records = []
    for dataset_key, grp in df.groupby("dataset_key"):
        for metric in PRIMARY_METRICS:
            if metric not in grp.columns:
                continue
            ranked = grp[["encoder", metric]].copy().dropna(subset=[metric])
            ranked["rank"] = ranked[metric].rank(ascending=False, method="average")
            for _, row in ranked.iterrows():
                records.append({
                    "encoder":     row["encoder"],
                    "metric":      metric,
                    "rank":        row["rank"],
                    "dataset_key": dataset_key,
                })
    rank_df = pd.DataFrame(records)
    mean_ranks = rank_df.groupby(["encoder", "metric"])["rank"].mean().reset_index()
    mean_ranks.columns = ["encoder", "metric", "mean_rank"]
    return mean_ranks


def _panel_heatmap(ax: plt.Axes, mean_ranks: pd.DataFrame) -> None:
    """Panel 1: encoder mean-rank heatmap with family colour bands."""
    pivot = mean_ranks.pivot(index="encoder", columns="metric", values="mean_rank")
    # Ensure consistent column order
    cols = [m for m in PRIMARY_METRICS if m in pivot.columns]
    pivot = pivot[cols]

    # Sort encoders: attention first (sorted by mean rank), then MP
    def _sort_key(enc: str) -> tuple:
        family = 0 if enc in ATTENTION_ENCODERS else 1
        return (family, pivot.loc[enc].mean() if enc in pivot.index else 999)

    ordered = sorted(pivot.index.tolist(), key=_sort_key)
    pivot = pivot.reindex(ordered)

    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=1,
                   vmax=max(data.max(), len(ordered)))

    # Column / row labels
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=8)

    # Cell annotations
    for i in range(len(ordered)):
        for j in range(len(cols)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v > data.max() * 0.65 else "black")

    # Highlight attention-family rows with a left-side band
    for i, enc in enumerate(ordered):
        colour = C_ATTENTION if enc in ATTENTION_ENCODERS else C_MP
        rect = mpatches.FancyBboxPatch(
            (-0.45, i - 0.45), 0.3, 0.9,
            boxstyle="square,pad=0",
            facecolor=colour, edgecolor="none", transform=ax.transData, clip_on=False,
        )
        ax.add_patch(rect)

    # Legend patches
    attn_patch = mpatches.Patch(color=C_ATTENTION, label="Attention family")
    mp_patch   = mpatches.Patch(color=C_MP,        label="Message-passing")
    ax.legend(handles=[attn_patch, mp_patch], loc="upper right", frameon=False,
              fontsize=7, bbox_to_anchor=(1.0, -0.15))

    plt.colorbar(im, ax=ax, shrink=0.6, label="Mean rank (lower = better)")
    ax.set_title("A  Encoder mean rank (12 encoders × 4 metrics)")


def _panel_family_bars(ax: plt.Axes, df: pd.DataFrame, stats_path: Path | None) -> None:
    """Panel 2: attention vs MP mean rank + Wilcoxon p overlay."""
    # Compute family mean ranks across all metrics
    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue

    # Build a per-dataset per-family summary
    has_family = "family" in df.columns
    if not has_family:
        df = df.copy()
        df["family"] = df["encoder"].apply(
            lambda e: "attention" if e in ATTENTION_ENCODERS else "message-passing"
        )

    # Mean rank per encoder across datasets, then average within family
    rank_df = _compute_mean_ranks(df)
    rank_df["family"] = rank_df["encoder"].apply(
        lambda e: "attention" if e in ATTENTION_ENCODERS else "message-passing"
    )
    family_metric = rank_df.groupby(["family", "metric"])["mean_rank"].mean().reset_index()

    metrics_present = [m for m in PRIMARY_METRICS if m in family_metric["metric"].values]
    x = np.arange(len(metrics_present))
    width = 0.3

    for i, fam in enumerate(["attention", "message-passing"]):
        color = "#D35400" if fam == "attention" else "#2471A3"
        vals = [family_metric.loc[
                    (family_metric["family"] == fam) & (family_metric["metric"] == m),
                    "mean_rank"].mean()
                for m in metrics_present]
        ax.bar(x + (i - 0.5) * width, vals, width, label=fam.capitalize(),
               color=color, alpha=0.85)

    # Overlay p-values if stats file exists
    if stats_path and stats_path.exists():
        stats = pd.read_csv(stats_path)
        for j, metric in enumerate(metrics_present):
            row = stats[stats["metric"] == metric] if "metric" in stats.columns else pd.DataFrame()
            if row.empty:
                continue
            p = row.iloc[0].get("p_value", None) or row.iloc[0].get("p_holm", None)
            if p is None:
                continue
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            y_top = max(
                family_metric.loc[family_metric["metric"] == metric, "mean_rank"].max(), 0
            ) + 0.3
            ax.text(x[j], y_top, stars, ha="center", va="bottom", fontsize=9,
                    color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_present, rotation=20, ha="right")
    ax.set_ylabel("Mean rank (lower = better)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("B  Attention vs message-passing family (mean rank)")
    ax.invert_yaxis()


def make_axisA_figure(sweep_dir: Path, stats_path: Path | None, out_dir: Path) -> None:
    """Build and save the Axis A encoder-ranking figure."""
    df = _load_encoder_results(sweep_dir)
    log.info("Loaded %d rows from %s", len(df), sweep_dir)

    mean_ranks = _compute_mean_ranks(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("scCCVGBen Axis A — Encoder Ranking", fontsize=12,
                 fontweight="bold")

    _panel_heatmap(ax1, mean_ranks)
    _panel_family_bars(ax2, df, stats_path)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"fig_axisA_encoder_ranking.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    repo = Path(__file__).parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", type=Path,
                   default=repo / "results" / "encoder_sweep")
    p.add_argument("--stats-csv", type=Path,
                   default=repo / "results" / "stats" / "axisA_attention_vs_mp.csv")
    p.add_argument("--out-dir", type=Path, default=repo / "figures")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    stats = args.stats_csv if args.stats_csv.exists() else None
    make_axisA_figure(args.sweep_dir, stats, args.out_dir)
