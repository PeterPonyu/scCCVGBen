"""Schema bridge between CG_dl_merged/ (scRNA) and CG_atacs/tables/ (scATAC) result CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical column order (subset used for reordering)
_CANONICAL_ORDER = [
    "method", "ASW", "DAV", "CAL", "COR",
    "distance_correlation_umap", "Q_local_umap", "Q_global_umap",
    "K_max_umap", "overall_quality_umap",
    "distance_correlation_tsne", "Q_local_tsne", "Q_global_tsne",
    "K_max_tsne", "overall_quality_tsne",
    "manifold_dimensionality_intrin", "spectral_decay_rate_intrin",
    "participation_ratio_intrin", "anisotropy_score_intrin",
    "trajectory_directionality_intrin", "noise_resilience_intrin",
    "core_quality_intrin", "overall_quality_intrin",
    "data_type_intrin", "interpretation_intrin",
    "NMI", "ARI",
]


def load_reused_csv(path: Path, modality: str) -> pd.DataFrame:
    """Load a reused result CSV and return canonical schema.

    Columns = [method, ASW, DAV, CAL, COR, ...intrin..., NMI, ARI]

    The scRNA path (CG_dl_merged) has 'method' as a named first column with NMI/ARI last.
    The scATAC path (CG_atacs/tables) has an unnamed first column (method as row value
    like 'LSI') with NMI/ARI second/third.

    This function detects format and normalises to canonical column order.
    """
    df = pd.read_csv(path)

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
