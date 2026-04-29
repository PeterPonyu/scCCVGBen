"""US-602: Outlier audit across reconciled scRNA/scATAC and encoder-sweep CSVs.

Outputs:
  data/metric_outliers_2026-04-28.csv
  .omc/research/metric-outliers-2026-04-28.md
  .omc/research/rerun-shortlist.txt
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from scccvgben.figures._long_form import melt_reconciled, melt_sweep

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SCRNA_DIR = ROOT / "results" / "reconciled" / "scrna"
SCATAC_DIR = ROOT / "results" / "reconciled" / "scatac"
SWEEP_DIR = ROOT / "results" / "encoder_sweep"
OUT_CSV = ROOT / "data" / "metric_outliers_2026-04-28.csv"
OUT_MD = ROOT / ".omc" / "research" / "metric-outliers-2026-04-28.md"
OUT_TXT = ROOT / ".omc" / "research" / "rerun-shortlist.txt"

# ---------------------------------------------------------------------------
# Biology-exclusion list: GSE115571_LPSMmDev UMAP/t-SNE neighbourhood metrics
# are known real biology, not a training failure.
# ---------------------------------------------------------------------------
BIOLOGY_METRICS = {
    "distance_correlation_umap", "Q_local_umap", "Q_global_umap",
    "overall_quality_umap", "distance_correlation_tsne", "Q_local_tsne",
    "Q_global_tsne", "overall_quality_tsne",
}
BIOLOGY_DATASET = "GSE115571_LPSMmDev"


def _is_biology(row: pd.Series) -> bool:
    return (
        str(row["dataset"]).startswith(BIOLOGY_DATASET)
        and row["metric"] in BIOLOGY_METRICS
    )


def _is_undertrained(row: pd.Series) -> bool:
    return str(row["dataset"]).startswith("GSE128033") and row["method"] == "scCCVGBen_GAT"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all() -> pd.DataFrame:
    frames = []
    for modality, path in [("scrna", SCRNA_DIR), ("scatac", SCATAC_DIR)]:
        df = melt_reconciled(path, modality)
        # keep NaN values too — we need to flag them
        df_raw = _reload_with_nans(path, modality)
        frames.append(df_raw)

    # encoder sweep (already wide, use melt_sweep — drops NaNs internally)
    sweep_raw = _reload_sweep_with_nans(SWEEP_DIR)
    frames.append(sweep_raw)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"dataset_id": "dataset"})
    return combined[["dataset", "dataset_name", "modality", "method", "metric", "value"]]


def _reload_with_nans(reconciled_dir: Path, modality: str) -> pd.DataFrame:
    """Like melt_reconciled but preserves NaN values instead of dropping them."""
    from scccvgben.data.result_csv_normalizer import load_reused_csv
    from scccvgben.figures.metrics import NUMERIC_METRICS
    from scccvgben.figures._long_form import dataset_key_from_result_stem

    keep = NUMERIC_METRICS
    frames = []
    for csv_path in sorted(reconciled_dir.glob("*.csv")):
        wide = load_reused_csv(csv_path, modality=modality)
        if "method" not in wide.columns:
            continue
        present = [m for m in keep if m in wide.columns]
        if not present:
            continue
        sub = wide[["method", *present]].copy()
        long = sub.melt(id_vars="method", var_name="metric", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        # do NOT drop NaN — we want them flagged
        long["dataset_id"] = dataset_key_from_result_stem(csv_path.stem)
        long["dataset_name"] = csv_path.stem
        long["modality"] = modality
        frames.append(long)

    if not frames:
        return pd.DataFrame(columns=["dataset_id", "dataset_name", "modality", "method", "metric", "value"])
    out = pd.concat(frames, ignore_index=True)
    return out[["dataset_id", "dataset_name", "modality", "method", "metric", "value"]]


def _reload_sweep_with_nans(sweep_dir: Path) -> pd.DataFrame:
    """Like melt_sweep but preserves NaN."""
    from scccvgben.figures.metrics import NUMERIC_METRICS
    from scccvgben.figures._long_form import dataset_key_from_result_stem

    keep = NUMERIC_METRICS
    frames = []
    for csv_path in sorted(sweep_dir.glob("*.csv")):
        wide = pd.read_csv(csv_path)
        if "method" not in wide.columns:
            continue
        present = [m for m in keep if m in wide.columns]
        if not present:
            continue
        sub = wide[["method", *present]].copy()
        long = sub.melt(id_vars="method", var_name="metric", value_name="value")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long["dataset_id"] = dataset_key_from_result_stem(csv_path.stem)
        long["dataset_name"] = csv_path.stem
        long["modality"] = "scrna"  # encoder sweep is all scRNA
        frames.append(long)

    if not frames:
        return pd.DataFrame(columns=["dataset_id", "dataset_name", "modality", "method", "metric", "value"])
    out = pd.concat(frames, ignore_index=True)
    return out[["dataset_id", "dataset_name", "modality", "method", "metric", "value"]]


# ---------------------------------------------------------------------------
# Z-score computation per (modality, method, metric) group
# ---------------------------------------------------------------------------

def compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Add z_score column; NaN rows get z_score=NaN."""
    df = df.copy()
    df["z_score"] = np.nan

    valid = df["value"].notna()
    for (modality, method, metric), grp_idx in df[valid].groupby(
        ["modality", "method", "metric"]
    ).groups.items():
        vals = df.loc[grp_idx, "value"]
        std = vals.std(ddof=1)
        mean = vals.mean()
        if std > 0:
            df.loc[grp_idx, "z_score"] = (vals - mean) / std
        else:
            df.loc[grp_idx, "z_score"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Flag all-zero rows: ≥3 metrics simultaneously zero for same (dataset, method)
# ---------------------------------------------------------------------------

def flag_all_zero(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series: True if this row belongs to a (dataset,method) pair
    that has ≥3 metric values == 0."""
    zero_counts = (
        df[df["value"].notna() & (df["value"] == 0)]
        .groupby(["dataset", "method"])
        .size()
        .rename("zero_count")
        .reset_index()
    )
    pairs_with_all_zero = set(
        zip(
            zero_counts.loc[zero_counts["zero_count"] >= 3, "dataset"],
            zero_counts.loc[zero_counts["zero_count"] >= 3, "method"],
        )
    )
    return df.apply(
        lambda r: (r["dataset"], r["method"]) in pairs_with_all_zero, axis=1
    )


# ---------------------------------------------------------------------------
# Classify each row
# ---------------------------------------------------------------------------

def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    is_nan = df["value"].isna()
    is_all_zero = flag_all_zero(df)
    abs_z = df["z_score"].abs()

    # biology exclusion
    bio_mask = df.apply(_is_biology, axis=1)
    # undertrained
    under_mask = df.apply(_is_undertrained, axis=1)

    severity = pd.Series("ok", index=df.index)
    likely_cause = pd.Series("", index=df.index)
    action = pd.Series("skip", index=df.index)

    # low: |z| >= 2 and < 3 (informational only)
    low_mask = (~is_nan) & (~is_all_zero) & (abs_z >= 2) & (abs_z < 3) & (~bio_mask)
    severity[low_mask] = "low"
    likely_cause[low_mask] = "extreme-z"
    action[low_mask] = "skip"

    # medium: |z| >= 3 (not biology)
    med_mask = (~is_nan) & (~is_all_zero) & (abs_z >= 3) & (~bio_mask) & (~under_mask)
    severity[med_mask] = "medium"
    likely_cause[med_mask] = "extreme-z"
    action[med_mask] = "rerun"

    # high: NaN
    severity[is_nan] = "high"
    likely_cause[is_nan] = "nan"
    action[is_nan] = "rerun"

    # high: all-zero
    zero_not_nan = is_all_zero & (~is_nan)
    severity[zero_not_nan] = "high"
    likely_cause[zero_not_nan] = "all-zero"
    action[zero_not_nan] = "rerun"

    # biology override
    severity[bio_mask] = "medium"
    likely_cause[bio_mask] = "biology"
    action[bio_mask] = "biology"

    # undertrained override
    severity[under_mask] = "medium"
    likely_cause[under_mask] = "undertrained"
    action[under_mask] = "rerun"

    df["severity"] = severity
    df["likely_cause"] = likely_cause
    df["action"] = action
    return df


# ---------------------------------------------------------------------------
# Markdown helpers (no tabulate dependency)
# ---------------------------------------------------------------------------

def _df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table without tabulate."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if not np.isnan(v) else "nan")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("Loading data...")
    raw = load_all()
    print(f"  {len(raw):,} rows loaded from {raw['dataset'].nunique()} datasets")

    print("Computing z-scores...")
    df = compute_zscores(raw)

    print("Classifying outliers...")
    df = classify(df)

    # Only retain rows that are flagged (not "ok")
    flagged = df[df["severity"] != "ok"].copy()

    # Build final output columns
    out = flagged[["dataset", "method", "metric", "value", "z_score",
                   "severity", "likely_cause", "action"]].copy()
    out = out.sort_values(["severity", "z_score"], ascending=[True, True],
                          key=lambda s: s if s.name != "severity"
                          else s.map({"high": 0, "medium": 1, "low": 2}))

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"  Wrote {OUT_CSV}")

    # Aggregate counts
    n_total_pairs = raw.groupby(["dataset", "method"]).ngroups
    n_nan = int((raw["value"].isna()).sum())
    n_all_zero_rows = int(flag_all_zero(raw).sum())
    n_extreme_z = int((flagged["likely_cause"] == "extreme-z").sum())
    n_rerun = int((flagged["action"] == "rerun").sum())
    rerun_pairs = (
        flagged[flagged["action"] == "rerun"]
        .groupby(["dataset", "method"])
        .ngroups
    )

    # Top-30 by severity then abs z
    top30 = (
        out.dropna(subset=["z_score"])
        .assign(_absz=lambda d: d["z_score"].abs())
        .sort_values(["severity", "_absz"],
                     ascending=[True, False],
                     key=lambda s: s if s.name != "severity"
                     else s.map({"high": 0, "medium": 1, "low": 2}))
        .drop(columns="_absz")
        .head(30)
    )
    # include NaN rows (no z) in top-30 high
    nan_rows = out[out["likely_cause"] == "nan"].head(30 - len(top30))
    top30 = pd.concat([top30, nan_rows]).head(30)

    # Markdown report
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Metric Outlier Audit — 2026-04-28",
        "",
        "## Aggregate counts",
        "",
        f"| Statistic | Count |",
        f"|-----------|-------|",
        f"| Total (dataset, method) pairs scanned | {n_total_pairs:,} |",
        f"| NaN metric values | {n_nan:,} |",
        f"| All-zero rows (≥3 zeros in same pair) | {n_all_zero_rows:,} |",
        f"| Rows with \\|z\\| ≥ 3 | {n_extreme_z:,} |",
        f"| action=rerun rows | {n_rerun:,} |",
        f"| action=rerun unique (dataset, method) pairs | {rerun_pairs:,} |",
        "",
        "## Top-30 most concerning rows",
        "",
        _df_to_md(top30),
        "",
        "## Biology exclusion note",
        "",
        f"Dataset **{BIOLOGY_DATASET}** (microglia LPS response) shows consistently "
        f"low neighbourhood-preservation scores on UMAP and t-SNE panels. These are "
        f"real biology: the activated microglia transcriptome is genuinely dissimilar "
        f"from resting cells, so neighbourhood distances are large. The following "
        f"metrics for this dataset are excluded from the rerun shortlist and classified "
        f"as `action=biology`: {', '.join(sorted(BIOLOGY_METRICS))}.",
        "",
    ]
    OUT_MD.write_text("\n".join(md_lines))
    print(f"  Wrote {OUT_MD}")

    # Rerun shortlist
    rerun_df = flagged[flagged["action"] == "rerun"][["dataset", "method", "metric"]]
    lines = [f"{r.dataset} {r.method} {r.metric}" for r in rerun_df.itertuples()]
    OUT_TXT.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"  Wrote {OUT_TXT} ({len(lines)} lines)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total pairs: {n_total_pairs:,} | NaN: {n_nan} | All-zero rows: {n_all_zero_rows} | action=rerun: {n_rerun}")


if __name__ == "__main__":
    main()
