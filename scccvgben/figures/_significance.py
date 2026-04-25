"""Pair selection for significance brackets.

Given a long DataFrame, run paired Wilcoxon (per-dataset paired) between a
reference method and every other method for one metric, Holm-correct across
the family, return the top-k most significant pairs.

Edge cases:
- n_pairs < 3: returns empty list (Wilcoxon under-determined).
- All-zero diffs: p_raw = 1.0.
- scipy raises (e.g., identical samples): wrapped, p_raw = 1.0.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _holm_correct(p_raws: list[float]) -> list[float]:
    order = np.argsort(p_raws)
    m_total = len(p_raws)
    p_holm = [1.0] * m_total
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = float(p_raws[idx]) * (m_total - rank)
        corrected = max(corrected, running_max)
        corrected = min(corrected, 1.0)
        running_max = corrected
        p_holm[idx] = corrected
    return p_holm


def select_significance_pairs(
    long_df: pd.DataFrame,
    metric: str,
    reference_method: str,
    *,
    group_col: str = "method",
    pair_col: str = "dataset_id",
    top_k: int = 3,
    alpha: float = 0.05,
) -> list[tuple[str, str, float]]:
    """Return up to top_k significant pairs (reference, other, p_holm).

    Filter to one metric, pivot by pair_col so each method has aligned
    per-dataset values, drop rows with NaN in reference or candidate,
    run paired Wilcoxon, Holm-correct, select.
    """
    sub = long_df.loc[long_df["metric"] == metric, [pair_col, group_col, "value"]]
    if sub.empty:
        return []

    pivot = sub.pivot_table(
        index=pair_col, columns=group_col, values="value", aggfunc="first"
    )
    if reference_method not in pivot.columns:
        return []
    others = [m for m in pivot.columns if m != reference_method]
    if not others:
        return []

    candidates: list[tuple[str, float]] = []
    p_raws: list[float] = []
    for other in others:
        paired = pivot[[reference_method, other]].dropna()
        if len(paired) < 3:
            p_raws.append(1.0)
            candidates.append((other, 1.0))
            continue
        ref = paired[reference_method].to_numpy()
        oth = paired[other].to_numpy()
        diff = ref - oth
        if np.all(diff == 0):
            p_raws.append(1.0)
            candidates.append((other, 1.0))
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p_raw = wilcoxon(ref, oth, alternative="two-sided", zero_method="wilcox")
            p_raws.append(float(p_raw))
            candidates.append((other, float(p_raw)))
        except (ValueError, RuntimeError):
            p_raws.append(1.0)
            candidates.append((other, 1.0))

    p_holm = _holm_correct(p_raws)
    enriched = [
        (reference_method, candidates[i][0], p_holm[i])
        for i in range(len(candidates))
    ]
    significant = [t for t in enriched if t[2] < alpha]
    significant.sort(key=lambda t: t[2])
    return significant[:top_k]
