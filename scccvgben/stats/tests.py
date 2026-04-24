"""Statistical tests for benchmark comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def wilcoxon_signed_rank_with_holm(
    samples_a: dict[str, np.ndarray],
    samples_b: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test per metric, Holm-Bonferroni corrected.

    Parameters
    ----------
    samples_a, samples_b : dicts mapping metric_name -> 1-D array of scores
        (paired; same length per metric)
    alpha : family-wise error rate

    Returns DataFrame with columns: metric, statistic, p_raw, p_holm, significant.
    """
    metrics = sorted(set(samples_a) & set(samples_b))
    rows = []
    for m in metrics:
        a, b = np.asarray(samples_a[m]), np.asarray(samples_b[m])
        diff = a - b
        if np.all(diff == 0):
            rows.append({"metric": m, "statistic": np.nan, "p_raw": 1.0})
            continue
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        rows.append({"metric": m, "statistic": float(stat), "p_raw": float(p)})

    df = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)

    # Holm-Bonferroni correction
    m_total = len(df)
    p_holm = []
    max_so_far = 0.0
    for rank, p_raw in enumerate(df["p_raw"]):
        corrected = p_raw * (m_total - rank)
        corrected = max(corrected, max_so_far)
        corrected = min(corrected, 1.0)
        max_so_far = corrected
        p_holm.append(corrected)

    df["p_holm"] = p_holm
    df["significant"] = df["p_holm"] < alpha
    return df[["metric", "statistic", "p_raw", "p_holm", "significant"]]


def cliff_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size in [-1, 1].

    Positive: a tends to be larger than b.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    dominance = sum(
        (1 if ai > bi else -1 if ai < bi else 0)
        for ai in a
        for bi in b
    )
    return dominance / (n_a * n_b)


def coefficient_of_variation(x: np.ndarray) -> float:
    """Coefficient of variation: std / mean."""
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    if mu == 0:
        return np.nan
    return float(np.std(x) / mu)


def attention_vs_mp_test(
    encoder_results: pd.DataFrame,
    metrics: list[str],
    historical_gat_rows: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper for Axis A claim: attention vs message-passing encoders.

    Parameters
    ----------
    encoder_results : DataFrame with columns ['encoder', 'family', *metrics, ...]
        'family' should be 'attention' or 'message-passing'.
    metrics         : list of metric column names to test.
    historical_gat_rows : if True, rows where family=='attention' and there is
        only one unique encoder (symlinked historical GAT runs) are excluded
        before testing to avoid pseudoreplication.

    Returns DataFrame from wilcoxon_signed_rank_with_holm comparing
    attention-family vs message-passing-family scores.
    """
    df = encoder_results.copy()

    if historical_gat_rows:
        attn_encoders = df.loc[df["family"] == "attention", "encoder"].unique()
        if len(attn_encoders) == 1:
            df = df[df["family"] != "attention"]

    attn = df[df["family"] == "attention"]
    mp = df[df["family"] == "message-passing"]

    samples_a: dict[str, np.ndarray] = {}
    samples_b: dict[str, np.ndarray] = {}

    for m in metrics:
        if m in df.columns:
            # Use all values from each group; pair by position (min length)
            a_vals = attn[m].dropna().values
            b_vals = mp[m].dropna().values
            n = min(len(a_vals), len(b_vals))
            if n >= 2:
                samples_a[m] = a_vals[:n]
                samples_b[m] = b_vals[:n]

    if not samples_a:
        return pd.DataFrame(columns=["metric", "statistic", "p_raw", "p_holm", "significant"])

    return wilcoxon_signed_rank_with_holm(samples_a, samples_b)
