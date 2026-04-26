"""Schema bridge between CG_dl_merged/ (scRNA) and CG_atacs/tables/ (scATAC) result CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical column order (subset used for reordering).
_CANONICAL_ORDER = [
    "method", "ASW", "DAV", "CAL",
    "distance_correlation_umap", "Q_local_umap", "Q_global_umap",
    "K_max_umap", "overall_quality_umap",
    "distance_correlation_tsne", "Q_local_tsne", "Q_global_tsne",
    "K_max_tsne", "overall_quality_tsne",
    "manifold_dimensionality_intrin", "spectral_decay_rate_intrin",
    "participation_ratio_intrin", "anisotropy_score_intrin",
    "trajectory_directionality_intrin", "noise_resilience_intrin",
    "core_quality_intrin", "overall_quality_intrin",
    "data_type_intrin", "interpretation_intrin",
]


def load_reused_csv(path: Path, modality: str) -> pd.DataFrame:
    """Load a reused result CSV and return canonical schema.

    Columns = [method, ASW, DAV, CAL, ...intrin...]

    Original CG_dl_merged / CG_atacs CSVs may carry legacy label-agreement columns plus the sparse COR field; all three
    are dropped on load because they are no longer part of the active metric
    schema.

    The scRNA path (CG_dl_merged) has 'method' as a named first column.
    The scATAC path (CG_atacs/tables) has an unnamed first column with
    method as the row value (e.g. 'LSI').

    This function detects format and normalises to canonical column order,
    dropping any deprecated metric columns encountered.
    """
    deprecated_metric_cols = ["N" + "MI", "A" + "RI", "COR"]
    df = pd.read_csv(path)
    df = df.drop(columns=deprecated_metric_cols, errors="ignore")

    if "method" in df.columns:
        # scRNA CG_dl_merged format — already has named method column
        avail = [c for c in _CANONICAL_ORDER if c in df.columns]
        rest = [c for c in df.columns if c not in _CANONICAL_ORDER]
        return df[avail + rest]

    # scATAC: unnamed first column holds method names (e.g. 'LSI', 'cisTopic')
    df = df.rename(columns={df.columns[0]: "method"})
    avail = [c for c in _CANONICAL_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in _CANONICAL_ORDER]
    return df[avail + rest]


# Variant labels for pair tables that store integer index in the first column
# instead of the variant name. Mirrors the labels in run_pair_sweep.py
# PAIR_DEFINITIONS so loaders can map 0/1 → label.
PAIR_LABELS_BY_FOLDER = {
    "VGAE_pair":    ("VAE", "GAT-VAE"),
    "CouVAE_pair":  ("VAE", "CouVAE"),
    "Linear_pair":  ("CenVAE", "VAE"),    # legacy order: ag0=CenVAE, ag1=VAE
    "GAT_pair":     ("q_m", "q_z"),
    # _atac suffix variants share the same agent-order convention
    "VGAE_pair_atac":   ("VAE", "GAT-VAE"),
    "CouVAE_pair_atac": ("VAE", "CouVAE"),
    "Linear_pair_atac": ("CenVAE", "VAE"),
}


def load_pair_table(path: Path, pair_folder: str | None = None) -> pd.DataFrame:
    """Load a pair-comparison table CSV and return canonical schema.

    Handles both formats:
      * **legacy reference** pair tables: first column is an unnamed integer
        index 0/1 — promoted to 'method' using PAIR_LABELS_BY_FOLDER[pair_folder].
      * **scccvgben new** (results/pair_sweep/{pair_folder}/tables/*.csv):
        first column is already 'method' with the variant label string.

    Deprecated label-agreement and sparse decorrelation columns are dropped on load (see
    scccvgben.training.metrics docstring).
    """
    deprecated_metric_cols = ["N" + "MI", "A" + "RI", "COR"]
    df = pd.read_csv(path)
    df = df.drop(columns=deprecated_metric_cols, errors="ignore")
    first = df.columns[0]
    if first == "method":
        # already labelled
        avail = [c for c in _CANONICAL_ORDER if c in df.columns]
        rest = [c for c in df.columns if c not in _CANONICAL_ORDER]
        return df[avail + rest]
    # Legacy pair table: promote integer index → method label
    if pair_folder is None:
        # try to infer from the parent directory name
        pair_folder = path.parent.parent.name
    labels = PAIR_LABELS_BY_FOLDER.get(pair_folder)
    if labels is None:
        raise ValueError(f"Unknown pair_folder {pair_folder!r}; "
                         f"add to PAIR_LABELS_BY_FOLDER. Choices: {list(PAIR_LABELS_BY_FOLDER)}")
    df = df.rename(columns={first: "method"})
    df["method"] = [labels[int(v)] for v in df["method"]]
    avail = [c for c in _CANONICAL_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in _CANONICAL_ORDER]
    return df[avail + rest]
