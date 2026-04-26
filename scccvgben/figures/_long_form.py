"""Reconciled per-dataset CSV directories -> long DataFrame.

Long-form schema: [dataset_id, dataset_name, modality, method, metric, value].
Modality is first-class; consumers do not re-derive it from path.

Schema asymmetry between scrna and scatac reconciled CSVs is hidden by
delegating to scccvgben.data.result_csv_normalizer.load_reused_csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from scccvgben.data.result_csv_normalizer import load_reused_csv
from scccvgben.figures.metrics import NUMERIC_METRICS

DEFAULT_METRICS: tuple[str, ...] = NUMERIC_METRICS

_RESULT_PREFIXES = ("Can_", "Dev_", "Imm_", "Neu_", "Oth_", "ATA_", "sup_", "Gen_")


def dataset_key_from_result_stem(stem: str) -> str:
    """Map local result filenames back to benchmark_manifest filename_key values.

    Strategy: drop the trailing ``_df`` and the leading category prefix
    (``Can_`` / ``Dev_`` / ``Gen_`` / etc.). Anything between the category
    prefix and the GSE accession is kept verbatim so manifest entries with
    descriptive aliases (e.g. ``endo_GSE84133`` / ``hESC_GSE144024`` /
    ``ifnHSPC_GSE226824``) still match the corresponding reconciled file.
    Earlier versions auto-stripped everything up to the ``GSE``/``GSM``
    token whenever the token was not at index 0; that silently broke the
    three descriptive aliases above, dropping their data from
    ``filter_to_manifest`` and forcing the figures into ``PRELIMINARY``
    output (97/100 instead of 100/100).
    """
    key = stem
    if key.endswith("_df"):
        key = key[:-3]
    for prefix in _RESULT_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    return key


def filter_to_manifest(
    long_df: pd.DataFrame,
    manifest: Path | str,
    *,
    modality: str | None = None,
) -> pd.DataFrame:
    """Filter long-form results to the active 100+100 benchmark manifest.

    If the manifest is unavailable or no rows match, return the input unchanged
    so partial local experiments can still render explicitly preliminary figures.
    """
    manifest = Path(manifest)
    if long_df.empty or not manifest.exists():
        return long_df
    cols = ["filename_key", "modality"]
    meta = pd.read_csv(manifest, usecols=lambda c: c in cols)
    if modality is not None and "modality" in meta.columns:
        meta = meta[meta["modality"].astype(str).str.lower() == modality.lower()]
    keep = set(meta["filename_key"].dropna().astype(str))
    filtered = long_df[long_df["dataset_id"].astype(str).isin(keep)].copy()
    return filtered if not filtered.empty else long_df


def melt_reconciled(
    reconciled_dir: Path | str,
    modality: str,
    *,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Walk reconciled_dir/*.csv, normalise each, melt into long form.

    Returns columns: dataset_id, dataset_name, modality, method, metric, value.
    Non-numeric / missing metric values are dropped.
    """
    reconciled_dir = Path(reconciled_dir)
    if not reconciled_dir.is_dir():
        raise FileNotFoundError(f"reconciled_dir does not exist: {reconciled_dir}")

    keep = tuple(metrics) if metrics is not None else DEFAULT_METRICS
    frames: list[pd.DataFrame] = []
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
        long = long.dropna(subset=["value"])
        long["dataset_id"] = dataset_key_from_result_stem(csv_path.stem)
        long["dataset_name"] = csv_path.stem
        long["modality"] = modality
        frames.append(long)

    if not frames:
        return pd.DataFrame(
            columns=["dataset_id", "dataset_name", "modality", "method", "metric", "value"]
        )
    out = pd.concat(frames, ignore_index=True)
    return out[["dataset_id", "dataset_name", "modality", "method", "metric", "value"]]


def melt_sweep(
    sweep_dir: Path | str,
    modality: str,
    *,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Same shape as melt_reconciled, for results/encoder_sweep or graph_sweep.

    These CSVs already have a 'method' column header — no normaliser needed.
    """
    sweep_dir = Path(sweep_dir)
    if not sweep_dir.is_dir():
        raise FileNotFoundError(f"sweep_dir does not exist: {sweep_dir}")

    keep = tuple(metrics) if metrics is not None else DEFAULT_METRICS
    frames: list[pd.DataFrame] = []
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
        long = long.dropna(subset=["value"])
        long["dataset_id"] = dataset_key_from_result_stem(csv_path.stem)
        long["dataset_name"] = csv_path.stem
        long["modality"] = modality
        frames.append(long)

    if not frames:
        return pd.DataFrame(
            columns=["dataset_id", "dataset_name", "modality", "method", "metric", "value"]
        )
    out = pd.concat(frames, ignore_index=True)
    return out[["dataset_id", "dataset_name", "modality", "method", "metric", "value"]]
