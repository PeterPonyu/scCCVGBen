"""GO Biological Process enrichment per latent dimension.

Mirrors the protocol used in the legacy CCVGAE case-study notebooks
(``CCVGAE3_IR.ipynb`` cell 11 onwards): for each latent dimension take the
top-N most strongly correlated genes and run :func:`gseapy.enrich` against a
local MSigDB GMT file. Returns a long-form DataFrame the dotplot consumes.

Species selection is automatic: if any of the gene symbols look like
upper-case human symbols (e.g. ``ACTB``, ``CDKN2A``) we use the human GMT;
otherwise we fall back to mouse.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Local MSigDB symbol-set GMT files. Order matters — first existing file wins.
_GMT_HUMAN_CANDIDATES = (
    Path("/home/zeyufu/Downloads/msigdb/c5.go.bp.v2024.1.Hs.symbols.gmt"),
    Path("/home/zeyufu/Downloads/msigdb/c5.go.bp.v2023.2.Hs.symbols.gmt"),
)
_GMT_MOUSE_CANDIDATES = (
    Path("/home/zeyufu/Downloads/msigdb/m5.go.bp.v2024.1.Mm.symbols.gmt"),
    Path("/home/zeyufu/Downloads/msigdb/m5.go.bp.v2023.2.Mm.symbols.gmt"),
)


def _resolve_gmt(species: str) -> Path | None:
    candidates = _GMT_HUMAN_CANDIDATES if species == "human" else _GMT_MOUSE_CANDIDATES
    for p in candidates:
        if p.exists():
            return p
    return None


def _guess_species(genes: Iterable[str]) -> str:
    """Heuristic: human gene symbols are dominantly all-uppercase
    (e.g. ``CDKN2A``, ``ACTB``); mouse symbols are typically title-cased
    (e.g. ``Cdkn2a``, ``Actb``).
    """
    sample = [g for g in genes if g and len(g) > 1][:50]
    if not sample:
        return "human"
    upper_share = sum(1 for g in sample if g.isupper()) / len(sample)
    return "human" if upper_share > 0.5 else "mouse"


def go_bp_enrichment_per_dim(
    top_k_df: pd.DataFrame,
    *,
    n_genes_per_dim: int = 100,
    top_terms: int = 8,
    species: str | None = None,
) -> pd.DataFrame:
    """For each latent dim, run GO BP enrichment on the top-N correlated genes.

    Inputs
    ------
    top_k_df : DataFrame from :func:`top_k_genes_per_dim` with columns
        ``[dim, rank, gene, rho, abs_rho]``. The function expects the
        provided ``top_k_df`` to have at least ``n_genes_per_dim`` rows per
        dim (caller must ensure ``k >= n_genes_per_dim``); when fewer rows
        are present that dim simply uses what's available.
    n_genes_per_dim : top-K gene cutoff per dim (default 100, matching
        CCVGAE legacy ``head(100)``).
    top_terms : number of GO BP terms to keep per dim.
    species : ``"human"`` / ``"mouse"`` / ``None`` (auto-detect).

    Returns
    -------
    DataFrame with columns
    ``[dim, term, n_overlap, n_term, percent, padj, neg_log10_padj]``.
    Returns an empty DataFrame on any failure (gseapy missing, GMT missing,
    no genes), so callers can fall back to a placeholder panel.
    """
    if top_k_df is None or top_k_df.empty:
        return pd.DataFrame(columns=["dim","term","n_overlap","n_term","percent","padj","neg_log10_padj"])

    try:
        import gseapy as gp
    except ImportError:
        log.warning("gseapy not installed — skipping GO BP enrichment")
        return pd.DataFrame(columns=["dim","term","n_overlap","n_term","percent","padj","neg_log10_padj"])

    if species is None:
        species = _guess_species(top_k_df["gene"].astype(str))
    gmt = _resolve_gmt(species)
    if gmt is None:
        log.warning("no MSigDB GMT file found for species=%s — skipping enrichment", species)
        return pd.DataFrame(columns=["dim","term","n_overlap","n_term","percent","padj","neg_log10_padj"])

    rows = []
    for dim, sub in top_k_df.groupby("dim", sort=True):
        # Use the |rho|-sorted top N genes (already sorted by abs_rho desc in top_k_genes_per_dim)
        genes = sub.sort_values("abs_rho", ascending=False)["gene"].astype(str).head(n_genes_per_dim).tolist()
        if not genes:
            continue
        try:
            enres = gp.enrich(genes, gene_sets=str(gmt), no_plot=True, verbose=False)
            res = enres.res2d
        except Exception as exc:
            log.warning("dim %d enrich failed: %s", dim, exc)
            continue
        if res is None or len(res) == 0:
            continue

        # Adjusted P-value column name varies between gseapy versions — try both
        padj_col = "Adjusted P-value" if "Adjusted P-value" in res.columns else "P-value"
        res = res.sort_values(padj_col).head(top_terms)
        for _, r in res.iterrows():
            ovr = str(r.get("Overlap", "0/0"))
            try:
                a, b = ovr.split("/")
                n_ovr = int(a)
                n_term = int(b)
                pct = n_ovr / n_term if n_term else 0.0
            except Exception:
                n_ovr = 0
                n_term = 0
                pct = 0.0
            padj = float(r.get(padj_col, 1.0))
            rows.append({
                "dim": int(dim),
                "term": str(r.get("Term", "?")),
                "n_overlap": n_ovr,
                "n_term": n_term,
                "percent": pct,
                "padj": padj,
                "neg_log10_padj": float(-np.log10(max(padj, 1e-300))),
            })

    return pd.DataFrame(rows)
