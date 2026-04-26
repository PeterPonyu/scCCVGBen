"""Top-K gene correlations per latent dimension and latent self-correlation.

The legacy reference bio-validation reported only the top-1 correlated gene per
latent dim. This module returns the top-K (K configurable, default 5) so the
multi-panel figure layout can show whether a dimension corresponds to a
single gene or to a coherent gene module.

``latent_self_correlation`` exposes the off-diagonal correlation matrix used
to assess whether the latent dimensions are well-decorrelated. Same formula
as the vendored reference ``_calc_corr`` (see
``scccvgben.training.metrics.compute_latent_dimension_decorrelation``) but
returns the full matrix for visualisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def top_k_genes_per_dim(
    latent: np.ndarray,
    expression: np.ndarray | pd.DataFrame,
    gene_names: list[str] | None = None,
    k: int = 5,
    method: str = "spearman",
) -> pd.DataFrame:
    """Per latent dimension, return the K genes most strongly correlated with it.

    Parameters
    ----------
    latent : (N, L) ndarray
        Encoder output.
    expression : (N, G) ndarray or DataFrame
        Cell × gene matrix (preferably log-normalised).
    gene_names : list[str], optional
        Names matching ``expression`` columns. If ``expression`` is a
        DataFrame its columns are used.
    k : int
        Number of top genes per dim.
    method : ``"spearman"`` | ``"pearson"``
        Correlation method.

    Returns
    -------
    DataFrame with one row per (dim, rank) pair: columns
    ``["dim", "rank", "gene", "rho", "abs_rho"]``.
    """
    if isinstance(expression, pd.DataFrame):
        if gene_names is None:
            gene_names = list(expression.columns)
        X = expression.to_numpy()
    else:
        X = np.asarray(expression)
        if gene_names is None:
            gene_names = [f"gene_{j}" for j in range(X.shape[1])]
    Z = np.asarray(latent)

    if Z.shape[0] != X.shape[0]:
        raise ValueError(f"row mismatch: latent {Z.shape[0]} vs expression {X.shape[0]}")

    # Spearman across all (latent_dim, gene) pairs in one shot via ranks
    if method == "spearman":
        Zr = pd.DataFrame(Z).rank(axis=0).to_numpy()
        Xr = pd.DataFrame(X).rank(axis=0).to_numpy()
    elif method == "pearson":
        Zr, Xr = Z, X
    else:
        raise ValueError(f"unknown method {method!r}")

    # Standardise → cosine = correlation
    def _z(a):
        a = a - a.mean(axis=0, keepdims=True)
        s = a.std(axis=0, keepdims=True)

        s[s == 0] = 1.0
        return a / s
    Zn, Xn = _z(Zr), _z(Xr)
    n = Z.shape[0]
    corr = (Zn.T @ Xn) / n   # (L, G)

    rows = []
    for d in range(corr.shape[0]):
        order = np.argsort(-np.abs(corr[d]))[:k]
        for rank, j in enumerate(order):
            rows.append({
                "dim": d,
                "rank": rank,
                "gene": gene_names[j],
                "rho": float(corr[d, j]),
                "abs_rho": float(abs(corr[d, j])),
            })
    return pd.DataFrame(rows)


def latent_self_correlation(latent: np.ndarray) -> np.ndarray:
    """Symmetric (L, L) absolute Pearson correlation matrix of latent dims."""
    Z = np.asarray(latent)
    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError("latent must be 2-D with >=2 dims")
    return np.abs(np.corrcoef(Z.T))
