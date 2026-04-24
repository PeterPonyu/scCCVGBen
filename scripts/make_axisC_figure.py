"""Generate fig_axisC_baselines.{pdf,png}.

4-panel figure: ARI / NMI / ASW / distance_correlation_umap.
Each panel: paired-dataset point plot (CCVGAE vs each of the 13 baselines),
with significance asterisks from results/stats/axisC_ccvgae_vs_baselines.csv.

Loads both fresh baseline results and reused results via
scccvgben.data.result_csv_normalizer.load_reused_csv to handle schema mismatches.
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

PRIMARY_METRICS = ["ARI", "NMI", "ASW", "distance_correlation_umap"]
BASELINES = [
    "PCA", "KPCA", "ICA", "FA", "NMF", "TSVD",
    "DICL", "scVI", "DIPVAE", "InfoVAE", "TCVAE", "HighBetaVAE",
]
CCVGAE_METHOD = "CCVGAE"

C_CCVGAE   = "#D35400"
C_BASELINE = "#7F8C8D"
C_WIN      = "#27AE60"
C_LOSE     = "#E74C3C"

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


def _load_baselines_results(
    baselines_dir: Path,
    reused_scrna_dir: Path | None,
    reused_scatac_dir: Path | None,
) -> pd.DataFrame:
    """Load all baseline CSVs (fresh + reused) into a single DataFrame."""
    frames = []

    def _add_dir(d: Path, tag: str) -> None:
        if d is None or not d.exists():
            return
        for csv_f in d.glob("*.csv"):
            try:
                df = pd.read_csv(csv_f)
                df["dataset_key"] = csv_f.stem
                df["_source"] = tag
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load %s: %s", csv_f, exc)

    _add_dir(baselines_dir, "fresh")
    _add_dir(reused_scrna_dir, "reused_scrna")
    _add_dir(reused_scatac_dir, "reused_scatac")

    if not frames:
        raise FileNotFoundError(
            f"No baseline CSVs found in {baselines_dir}, "
            f"{reused_scrna_dir}, or {reused_scatac_dir}"
        )

    df = pd.concat(frames, ignore_index=True)
    # Normalize method column
    method_col = next(
        (c for c in df.columns if c.lower() in ("method", "baseline", "encoder")), None
    )
    if method_col and method_col != "method":
        df = df.rename(columns={method_col: "method"})
    return df


def _load_stats(stats_path: Path | None) -> pd.DataFrame:
    if stats_path and stats_path.exists():
        return pd.read_csv(stats_path)
    return pd.DataFrame()


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _panel_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    stats: pd.DataFrame,
) -> None:
    """One panel: CCVGAE vs each baseline, violin + strip."""
    if metric not in df.columns:
        ax.text(0.5, 0.5, f"{metric}\n(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title(metric)
        return

    ccvgae_vals = df[df["method"] == CCVGAE_METHOD].groupby("dataset_key")[metric].mean()
    baselines_present = [b for b in BASELINES if b in df["method"].values]

    positions = np.arange(len(baselines_present))
    diffs = []  # CCVGAE minus baseline per dataset, per baseline

    for bl in baselines_present:
        bl_vals = df[df["method"] == bl].groupby("dataset_key")[metric].mean()
        common  = ccvgae_vals.index.intersection(bl_vals.index)
        diff    = (ccvgae_vals.loc[common] - bl_vals.loc[common]).values
        diffs.append(diff)

    # Violin + strip
    vp = ax.violinplot(
        [d for d in diffs if len(d) > 0],
        positions=positions[:len([d for d in diffs if len(d) > 0])],
        showmedians=True, widths=0.5,
    )
    for pc in vp["bodies"]:
        pc.set_facecolor(C_CCVGAE)
        pc.set_alpha(0.4)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        if part in vp:
            vp[part].set_color("black")
            vp[part].set_linewidth(0.6)

    for i, diff in enumerate(diffs):
        if len(diff) == 0:
            continue
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(diff))
        colors = [C_WIN if v > 0 else C_LOSE for v in diff]
        ax.scatter(np.full(len(diff), i) + jitter, diff,
                   c=colors, s=4, alpha=0.5, zorder=3)

    # Reference line at 0
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

    # Significance asterisks
    if not stats.empty and "baseline" in stats.columns and "metric" in stats.columns:
        for i, bl in enumerate(baselines_present):
            row = stats[(stats["baseline"] == bl) & (stats["metric"] == metric)]
            if row.empty:
                continue
            p_col = next((c for c in ("p_holm", "p_value") if c in row.columns), None)
            if p_col is None:
                continue
            p = float(row.iloc[0][p_col])
            y_top = max(d.max() for d in diffs if len(d) > 0) * 1.05
            ax.text(i, y_top, _stars(p), ha="center", va="bottom", fontsize=7, color="#333")

    ax.set_xticks(range(len(baselines_present)))
    ax.set_xticklabels(baselines_present, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(f"ΔCCVGAE ({metric})")
    ax.set_title(metric)


def make_axisC_figure(
    baselines_dir: Path,
    reused_scrna_dir: Path | None,
    reused_scatac_dir: Path | None,
    stats_path: Path | None,
    out_dir: Path,
) -> None:
    """Build and save the Axis C baselines figure."""
    df = _load_baselines_results(baselines_dir, reused_scrna_dir, reused_scatac_dir)
    log.info("Loaded %d rows from baselines dirs", len(df))
    stats = _load_stats(stats_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("scCCVGBen Axis C — CCVGAE vs Baselines",
                 fontsize=12, fontweight="bold")

    for ax, metric in zip(axes.flat, PRIMARY_METRICS):
        _panel_metric(ax, df, metric, stats)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"fig_axisC_baselines.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        log.info("Saved %s", out_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    repo = Path(__file__).parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baselines-dir", type=Path,
                   default=repo / "results" / "baselines")
    p.add_argument("--reused-scrna-dir", type=Path,
                   default=repo / "workspace" / "reused_results" / "scrna_baselines")
    p.add_argument("--reused-scatac-dir", type=Path,
                   default=repo / "workspace" / "reused_results" / "scatac_baselines")
    p.add_argument("--stats-csv", type=Path,
                   default=repo / "results" / "stats" / "axisC_ccvgae_vs_baselines.csv")
    p.add_argument("--out-dir", type=Path, default=repo / "figures")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    make_axisC_figure(
        args.baselines_dir,
        args.reused_scrna_dir if args.reused_scrna_dir.exists() else None,
        args.reused_scatac_dir if args.reused_scatac_dir.exists() else None,
        args.stats_csv if args.stats_csv.exists() else None,
        args.out_dir,
    )
