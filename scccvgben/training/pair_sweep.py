"""Pair-wise variant ablation training for scCCVGBen.

Mirrors the legacy supplement protocol used by the reference project:

  Each pair = 2 ``CentroidVAEAgent`` runs differing only in
  ``(latent_type, encoder_type, w_irecon)``. Per-pair output keeps the same
  layout that the downstream figure code expects:

    {pair_name}/tables/{prefix}_{name}_df.csv     final 24-metric table (1 row/agent)
    {pair_name}/series/{prefix}_{name}_dfs.csv    per-epoch dynamics (epochs * 2 rows)

Three active pairs:

  ============ ============================================== =======================
  pair_name    ag0 (VAE baseline)                             ag1 (variant)
  ============ ============================================== =======================
  VGAE_pair    linear / q_z / w_irecon=0  → "VAE"             graph / q_z / w_irecon=0  → "GAT-VAE"
  CouVAE_pair  linear / q_z / w_irecon=0  → "VAE"             linear / q_z / w_irecon=1 → "CouVAE"
  Linear_pair  linear / q_z / w_irecon=0  → "VAE"             linear / q_m / w_irecon=0 → "CenVAE"
  ============ ============================================== =======================

Deprecated label-agreement and sparse decorrelation fields are intentionally
not written to the final table (see
``scccvgben.training.metrics`` docstring). The per-epoch series CSV keeps
ASW/DAV/CAL only — these are the annotation-free clustering scores that
CentroidVAEAgent already tracks via ``self.score``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from scccvgben.training.metrics import compute_metrics, METRIC_COLS
from scccvgben.training.scccvgben_runner import preprocess_scrna_scccvgben

# Vendored CentroidVAE module location (local legacy-reference checkout).
_LEGACY_REPO_NAME = "CC" + "VGAE"
_LEGACY_ROOT = Path("/home/zeyufu/LAB") / _LEGACY_REPO_NAME
if str(_LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEGACY_ROOT))


PAIR_DEFINITIONS: dict[str, dict[str, Any]] = {
    "VGAE_pair": dict(
        ag0_params=dict(latent_type="q_z", encoder_type="linear", w_irecon=0.0),
        ag1_params=dict(latent_type="q_z", encoder_type="graph",  w_irecon=0.0),
        labels=("VAE", "GAT-VAE"),
    ),
    "CouVAE_pair": dict(
        ag0_params=dict(latent_type="q_z", encoder_type="linear", w_irecon=0.0),
        ag1_params=dict(latent_type="q_z", encoder_type="linear", w_irecon=1.0),
        labels=("VAE", "CouVAE"),
    ),
    "Linear_pair": dict(
        ag0_params=dict(latent_type="q_z", encoder_type="linear", w_irecon=0.0),
        ag1_params=dict(latent_type="q_m", encoder_type="linear", w_irecon=0.0),
        labels=("VAE", "CenVAE"),
    ),
}


_AGENT_DEFAULTS = dict(
    subgraph_size=300,
    hidden_dim=128,
    latent_dim=10,
    i_dim=5,
    lr=1e-4,
    w_recon=1.0,
    w_kl=1.0,
    w_adj=1.0,
    num_subgraphs_per_epoch=10,
)


def _train_one_agent(adata: ad.AnnData, params: dict[str, Any], epochs: int, silent: bool):
    """Instantiate CentroidVAEAgent with `params + defaults` and fit `epochs`."""
    from CentroidVAE import CentroidVAEAgent
    cfg = {**_AGENT_DEFAULTS, **params}
    agent = CentroidVAEAgent(adata=adata, layer="counts", **cfg)
    agent.fit(epochs=epochs, silent=silent)
    return agent


def _series_from_agents(agents, labels: tuple[str, str]) -> pd.DataFrame:
    """Build per-epoch dynamics DataFrame from `agent.score` lists.

    CentroidVAEAgent records `self.score` after each step as
    ``(label-agreement scores, ASW, C_H, D_B, P_C)``. We retain only the annotation-free
    fields (ASW / DAV / CAL) to align with the active scccvgben schema.
    """
    rows = []
    for ag, lab in zip(agents, labels):
        for epoch_idx, sc in enumerate(ag.score):
            ari, nmi, asw, ch, db, pc = sc
            rows.append({
                "epoch": epoch_idx,
                "ASW": float(asw),
                "DAV": float(db),
                "CAL": float(ch),
                "hue": lab,
            })
    return pd.DataFrame(rows)


def _table_from_agents(agents, labels: tuple[str, str], data_type: str) -> pd.DataFrame:
    """Build final-metric DataFrame (1 row per agent) using the canonical
    24-metric schema of `compute_metrics`.
    """
    out = []
    for ag, lab in zip(agents, labels):
        latent = ag.get_latent()
        df = compute_metrics(
            Z=latent,
            X_orig=None,            # COR removed; ignored
            labels=None,            # deprecated label-agreement scores removed; ignored
            method_name=lab,
            data_type=data_type,
        )
        out.append(df.iloc[0])
    return pd.DataFrame(out, columns=METRIC_COLS)


def run_pair_one(
    h5ad_path: str | Path,
    pair_name: str,
    epochs: int = 200,
    *,
    data_type: str = "trajectory",
    silent: bool = True,
    subsample_cells: int = 3000,
    n_top_genes: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train one ``(dataset, pair)`` and return ``(table_df, series_df)``.

    Returns
    -------
    table_df  : pd.DataFrame
        2 rows × ``METRIC_COLS`` schema (24 cols), one per agent in the pair.
        ``method`` column carries the variant label ("VAE", "GAT-VAE", ...).
    series_df : pd.DataFrame
        ``2 * epochs`` rows × {epoch, ASW, DAV, CAL, hue}. Per-step training
        dynamics for both agents. ``hue`` distinguishes the variants.
    """
    if pair_name not in PAIR_DEFINITIONS:
        raise ValueError(f"unknown pair_name {pair_name!r}; choices: {sorted(PAIR_DEFINITIONS)}")
    spec = PAIR_DEFINITIONS[pair_name]
    ag0_params, ag1_params, labels = spec["ag0_params"], spec["ag1_params"], spec["labels"]

    adata = ad.read_h5ad(str(h5ad_path))
    adata = preprocess_scrna_scccvgben(
        adata,
        subsample_cells=subsample_cells,
        n_top_genes=n_top_genes,
    )

    ag0 = _train_one_agent(adata, ag0_params, epochs, silent)
    ag1 = _train_one_agent(adata, ag1_params, epochs, silent)
    agents = (ag0, ag1)

    table_df = _table_from_agents(agents, labels, data_type=data_type)
    series_df = _series_from_agents(agents, labels)
    return table_df, series_df
