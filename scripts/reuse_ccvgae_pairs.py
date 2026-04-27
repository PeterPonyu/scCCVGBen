#!/usr/bin/env python3
"""Reuse legacy CCVGAE pair_sweep results for Linear_pair + VGAE_pair.

Background
----------
The CCVGAE-revised supplement trained 4 pair variants on 55 scRNA datasets:

    /home/zeyufu/LAB/CCVGAE/CentroidVAE_results/{Linear,CouVAE,VGAE,GAT}_pair/
        tables/{prefix}_{name}_df.csv      ← 27-col legacy schema
        series/{prefix}_{name}_dfs.csv     ← 8-col legacy schema

scccvgben uses the SAME CentroidVAEAgent class but overrides ``i_dim=5``
(legacy uses the agent default ``i_dim=10``). ``i_dim`` only enters the loss
when ``w_irecon > 0``. Of the three scccvgben pairs:

    Linear_pair: ag0 w_irecon=0, ag1 w_irecon=0 → i_dim irrelevant — REUSE SAFE
    VGAE_pair  : ag0 w_irecon=0, ag1 w_irecon=0 → i_dim irrelevant — REUSE SAFE
    CouVAE_pair: ag0 w_irecon=0, ag1 w_irecon=1 → i_dim CHANGES outputs — DO NOT REUSE

This script reuses the SAFE two pairs only, transforming legacy CSVs into the
scccvgben canonical schema on the way in:

  1. Tables: drop NMI/ARI/COR; promote integer index 0/1 to ``method`` column
     using ``PAIR_LABELS_BY_FOLDER``.
  2. Series: drop NMI/ARI/COR; remap legacy ``hue`` values to scccvgben labels
     (e.g. ``q_m`` → ``CenVAE``).
  3. File naming: re-derive the category prefix using scccvgben's
     ``_resolve_prefix`` so the in-flight retrain's ``is_done`` check skips
     these datasets when matched.

Idempotent: skips any output file that already exists. The in-flight pair
sweep (PID 2463479) skips a (dataset, pair) when both tables/{}_df.csv and
series/{}_dfs.csv are present, so writing here directly causes the
background retrain to converge to scccvgben's full 100-manifest coverage
without redundant compute on the 45 CCVGAE-overlap datasets.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_pair_sweep import _resolve_prefix  # noqa: E402

LEGACY_ROOT = Path("/home/zeyufu/LAB/CCVGAE/CentroidVAE_results")
NEW_ROOT = REPO_ROOT / "results" / "pair_sweep"

# Pairs to reuse — i_dim doesn't affect outputs because both ag0 and ag1
# have w_irecon=0.
SAFE_PAIRS = ("Linear_pair", "VGAE_pair")

DEPRECATED_COLS = ("N" + "MI", "A" + "RI", "COR")

# Legacy series uses these hue values — we remap them to scccvgben labels.
SERIES_HUE_REMAP = {
    "Linear_pair": {"q_m": "CenVAE", "q_z": "VAE"},
    "VGAE_pair":   {"VAE": "VAE", "VGAE": "GAT-VAE"},
}

# Method labels for promoting legacy integer index 0/1 in tables. These match
# the order of ag0/ag1 in CCVGAE_supplement/run_centroidvae_supplement.py:
#   Linear_pair = (q_m, q_z)   → scccvgben labels: (CenVAE, VAE)
#   VGAE_pair   = (VAE, VGAE)  → scccvgben labels: (VAE, GAT-VAE)
TABLE_METHOD_LABELS = {
    "Linear_pair": ("CenVAE", "VAE"),
    "VGAE_pair":   ("VAE", "GAT-VAE"),
}


def _convert_table(legacy_csv: Path, pair: str) -> pd.DataFrame:
    df = pd.read_csv(legacy_csv)
    df = df.drop(columns=list(DEPRECATED_COLS), errors="ignore")
    first = df.columns[0]
    if first == "method":
        return df
    labels = TABLE_METHOD_LABELS[pair]
    df = df.rename(columns={first: "method"})
    df["method"] = [labels[int(v)] for v in df["method"]]
    cols = ["method"] + [c for c in df.columns if c != "method"]
    return df[cols]


def _convert_series(legacy_csv: Path, pair: str) -> pd.DataFrame:
    df = pd.read_csv(legacy_csv)
    df = df.drop(columns=list(DEPRECATED_COLS), errors="ignore")
    # Drop legacy unnamed index column if it survived
    first = df.columns[0]
    if first.startswith("Unnamed") or first == "":
        df = df.drop(columns=[first])
    # Bug in CCVGAE_supplement/run_centroidvae_supplement.py: per-epoch
    # `agent.score` is (ari, nmi, asw, ch, db, pc) but legacy code labels the
    # DataFrame columns as ('NMI','ARI','ASW','DAV','CAL','COR') — so the
    # legacy "DAV" column actually holds Calinski-Harabasz values and vice
    # versa. Verified empirically: legacy "DAV" range 134-11600 is CAL-like,
    # legacy "CAL" range 0.8-2.2 is DAV-like. Swap to restore correct labels.
    if "DAV" in df.columns and "CAL" in df.columns:
        df = df.rename(columns={"DAV": "_legacy_CAL", "CAL": "DAV"})
        df = df.rename(columns={"_legacy_CAL": "CAL"})
    if "hue" in df.columns:
        remap = SERIES_HUE_REMAP[pair]
        df["hue"] = df["hue"].astype(str).map(lambda v: remap.get(v, v))
    if "epoch" not in df.columns:
        df.insert(0, "epoch", df.groupby("hue").cumcount() if "hue" in df.columns else range(len(df)))
    keep = [c for c in ("epoch", "ASW", "DAV", "CAL", "hue") if c in df.columns]
    return df[keep]


def _rename_to_scccvgben(legacy_filename: str) -> str | None:
    """Map legacy filename (e.g. Dev_GSE120505_bloodAged_df.csv) to scccvgben
    naming where the prefix is re-derived from the stem via ``_resolve_prefix``.
    Returns the scccvgben filename, or None if no prefix recognised.
    """
    m = re.match(r"(Can|Gen|Dev|sup)_(.+)(_df\.csv|_dfs\.csv)$", legacy_filename)
    if not m:
        return None
    _, stem, suffix = m.groups()
    scc_prefix = _resolve_prefix(stem)
    return f"{scc_prefix}_{stem}{suffix}"


def main() -> int:
    n_table_new = n_table_skip = 0
    n_series_new = n_series_skip = 0

    for pair in SAFE_PAIRS:
        legacy_tables = sorted((LEGACY_ROOT / pair / "tables").glob("*_df.csv"))
        legacy_series = sorted((LEGACY_ROOT / pair / "series").glob("*_dfs.csv"))
        new_tables_dir = NEW_ROOT / pair / "tables"
        new_series_dir = NEW_ROOT / pair / "series"
        new_tables_dir.mkdir(parents=True, exist_ok=True)
        new_series_dir.mkdir(parents=True, exist_ok=True)

        for f in legacy_tables:
            new_name = _rename_to_scccvgben(f.name)
            if new_name is None:
                continue
            out = new_tables_dir / new_name
            if out.exists():
                n_table_skip += 1
                continue
            df = _convert_table(f, pair)
            df.to_csv(out, index=False)
            n_table_new += 1

        for f in legacy_series:
            new_name = _rename_to_scccvgben(f.name)
            if new_name is None:
                continue
            out = new_series_dir / new_name
            if out.exists():
                n_series_skip += 1
                continue
            df = _convert_series(f, pair)
            df.to_csv(out, index=False)
            n_series_new += 1

    print(f"Reused legacy CCVGAE pair_sweep results for {SAFE_PAIRS}:")
    print(f"  tables: +{n_table_new} new, {n_table_skip} skipped")
    print(f"  series: +{n_series_new} new, {n_series_skip} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
