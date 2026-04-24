"""Generate fig_axisB_graph_robustness.{pdf,png} from results/graph_sweep/*.csv.

CV bars across 5 graph constructions for ARI and NMI.
Reference line for best-baseline CV loaded from results/stats/axisB_graph_robustness.csv.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GRAPH_ORDER = ["kNN_euclidean", "kNN_cosine", "snn", "mutual_knn", "gaussian"]
PRIMARY_METRICS = ["ARI", "NMI"]
C_SCRNA  = "#D35400"
C_SCATAC = "#2471A3"

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


def _load_graph_results(sweep_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_f in sweep_dir.glob("*.csv"):
        df = pd.read_csv(csv_f)
        df["dataset_key"] = csv_f.stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSVs found in {sweep_dir}")
    return pd.concat(frames, ignore_index=True)


def _coefficient_of_variation(vals: np.ndarray) -> float:
    """CV = std / mean; returns 0 if mean == 0."""
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2 or vals.mean() == 0:
        return 0.0
    return float(vals.std() / abs(vals.mean()))


def make_axisB_figure(sweep_dir: Path, stats_path: Path | None, out_dir: Path) -> None:
    """Build and save the Axis B graph-robustness figure."""
    df = _load_graph_results(sweep_dir)
    log.info("Loaded %d rows from %s", len(df), sweep_dir)

    # Normalize graph column name
    graph_col = next((c for c in df.columns if c.lower() in ("graph", "graph_name")), None)
    if graph_col and graph_col != "graph":
        df = df.rename(columns={graph_col: "graph"})

    # Compute per-dataset CV for each (graph × metric) cell, then average across datasets
    cv_records = []
    for dataset_key, grp in df.groupby("dataset_key"):
        for metric in PRIMARY_METRICS:
            if metric not in grp.columns:
                continue
            # Group by graph, average metrics (should be 1 row per graph already)
            for graph, g2 in grp.groupby("graph"):
                cv_records.append({
                    "dataset_key": dataset_key,
                    "graph":       graph,
                    "metric":      metric,
                    "value":       g2[metric].mean(),
                })

    cv_df = pd.DataFrame(cv_records)

    # For each dataset, compute CV across graphs, then mean across datasets
    cv_agg = (
        cv_df.groupby(["dataset_key", "metric"])["value"]
        .apply(lambda x: _coefficient_of_variation(x.values))
        .reset_index(name="cv")
    )

    # Load baseline reference line
    ref_cv: dict[str, float] = {}
    if stats_path and stats_path.exists():
        ref = pd.read_csv(stats_path)
        for metric in PRIMARY_METRICS:
            row = ref[ref["metric"] == metric] if "metric" in ref.columns else pd.DataFrame()
            if not row.empty and "best_baseline_cv" in row.columns:
                ref_cv[metric] = float(row.iloc[0]["best_baseline_cv"])

    fig, axes = plt.subplots(1, len(PRIMARY_METRICS), figsize=(10, 5), sharey=False)
    if len(PRIMARY_METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, PRIMARY_METRICS):
        data = cv_agg[cv_agg["metric"] == metric]["cv"].dropna()
        mean_cv  = data.mean()
        # Also per-graph breakdown
        per_graph = []
        for graph in GRAPH_ORDER:
            g_data = cv_df[(cv_df["metric"] == metric) & (cv_df["graph"] == graph)]["value"]
            cv_val = _coefficient_of_variation(g_data.values)
            per_graph.append((graph, cv_val))

        graphs_present = [g for g, _ in per_graph if not np.isnan(_)]
        cv_vals = [cv for _, cv in per_graph if not np.isnan(cv)]

        x = np.arange(len(graphs_present))
        bars = ax.bar(x, cv_vals, color=C_SCRNA, alpha=0.8, width=0.5)

        # Reference line from best baseline
        if metric in ref_cv:
            ax.axhline(ref_cv[metric], color=C_SCATAC, linestyle="--", linewidth=1.2,
                       label=f"Best baseline CV ({ref_cv[metric]:.3f})")
            ax.legend(frameon=False, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(graphs_present, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Coefficient of variation (CV)" if metric == PRIMARY_METRICS[0] else "")
        ax.set_title(f"Graph robustness — {metric}")

        # Value annotations
        for bar, val in zip(bars, cv_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("scCCVGBen Axis B — Graph Construction Robustness",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"fig_axisB_graph_robustness.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    repo = Path(__file__).parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-dir", type=Path,
                   default=repo / "results" / "graph_sweep")
    p.add_argument("--stats-csv", type=Path,
                   default=repo / "results" / "stats" / "axisB_graph_robustness.csv")
    p.add_argument("--out-dir", type=Path, default=repo / "figures")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    stats = args.stats_csv if args.stats_csv.exists() else None
    make_axisB_figure(args.sweep_dir, stats, args.out_dir)
