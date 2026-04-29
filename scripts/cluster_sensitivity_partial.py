"""D5 partial: Spearman rank-correlation of method orderings across NMI, ARI, ASW.

Addresses reviewer 1.2 (clustering parameters) by showing that the ranking of
methods is stable across the three available summary metrics (ARI, NMI, ASW).
High Spearman r across these means the scCCVGBen advantage is not an artifact
of the particular metric used.

Usage:
    python scripts/cluster_sensitivity_partial.py

Output:
    results/cluster_sensitivity_partial_2026-04-28.csv
    (columns: dataset, r_ARI_NMI, r_ARI_ASW, r_NMI_ASW, n_methods)

Full D5 (Leiden / k-means / Louvain re-clustering on raw latents) is blocked
until latents are persisted by the training pipeline.  See
.omc/research/D5_D6_blocker-2026-04-28.md for the 2-line fix.
"""
from __future__ import annotations

import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
SCRNA_DIR = ROOT / "results" / "reconciled" / "scrna"
OUT_CSV = ROOT / "results" / "cluster_sensitivity_partial_2026-04-28.csv"

METRICS = ["ARI", "NMI", "ASW"]
METRIC_PAIRS = list(combinations(METRICS, 2))


# ---------------------------------------------------------------------------
# Per-dataset worker (safe for multiprocessing)
# ---------------------------------------------------------------------------

def _process_one(csv_path: Path) -> dict | None:
    """Compute Spearman r between metric rankings for one dataset CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Need at least 3 methods and all three metric columns present
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        return None

    # Drop rows where any metric is NaN
    sub = df[METRICS].dropna()
    n_methods = len(sub)
    if n_methods < 3:
        return None

    row: dict = {
        "dataset": csv_path.stem,
        "n_methods": n_methods,
    }
    for m1, m2 in METRIC_PAIRS:
        r, _ = spearmanr(sub[m1].values, sub[m2].values)
        row[f"r_{m1}_{m2}"] = round(float(r), 4)

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    csv_paths = sorted(SCRNA_DIR.glob("*_df.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found under {SCRNA_DIR}")

    print(f"Processing {len(csv_paths)} dataset CSVs with Pool(8)...")

    with Pool(processes=8) as pool:
        results = pool.map(_process_one, csv_paths)

    rows = [r for r in results if r is not None]
    if not rows:
        raise RuntimeError("No valid rows produced — check CSV structure.")

    out_df = pd.DataFrame(rows)
    out_df.sort_values("dataset", inplace=True)
    out_df.reset_index(drop=True, inplace=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    elapsed = time.time() - t0
    print(f"\nWrote {len(out_df)} rows -> {OUT_CSV}")
    print(f"Wall-clock time: {elapsed:.1f}s")

    # Summary stats
    print("\n--- Spearman r summary across datasets ---")
    for col in [f"r_{m1}_{m2}" for m1, m2 in METRIC_PAIRS]:
        vals = out_df[col].dropna()
        print(f"  {col}: mean={vals.mean():.3f}  SD={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")

    print("\nInterpretation:")
    print("  r > 0.9 across all pairs -> method ranking is metric-agnostic.")
    print("  (Full D5/D6 awaits latent persistence; see .omc/research/D5_D6_blocker-2026-04-28.md)")


if __name__ == "__main__":
    main()
