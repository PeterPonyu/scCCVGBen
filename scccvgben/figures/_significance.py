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


def compute_pair_effects(
    long_df: pd.DataFrame,
    metric: str,
    ref: str,
    other: str,
    *,
    group_col: str = "method",
    pair_col: str = "dataset_id",
    ci_alpha: float = 0.05,
    seed: int = 0,
) -> dict | None:
    """Effect size + CI + raw Wilcoxon p for one (ref, other) comparison.

    No top-k filter, no alpha gate — every requested comparison gets a row.
    Holm correction is intentionally NOT applied here: it must run at the
    family level (per-figure) by the caller via :func:`holm_correct_pairs`,
    because Holm depends on the full set of comparisons in the family.

    Sign convention: ``mean_diff = mean(ref - other)`` per dataset, where
    "ref" should be the proposed/improved method and "other" the baseline.
    For higher-is-better metrics (e.g. ASW), positive ``mean_diff`` means the
    reference method wins. For lower-is-better metrics (e.g. ``DAV``, see
    :data:`scccvgben.figures.metrics.LOWER_IS_BETTER`), positive ``mean_diff``
    actually means the reference method *loses* on that metric — callers that
    want a uniform "positive == improvement" reading must invert the sign for
    those metrics themselves. This helper does NOT auto-invert; it returns
    the raw paired difference so downstream rendering / formatting can
    decide.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-form metrics frame with columns including ``metric``, ``value``,
        ``group_col`` (default ``method``) and ``pair_col`` (default
        ``dataset_id``).
    metric : str
        The single metric to filter to (e.g. ``"ASW"``).
    ref, other : str
        Method labels in ``group_col``; ``mean(ref - other)`` is reported.
    group_col, pair_col : str
        Column names for the method label and the pairing key.
    ci_alpha : float
        Two-sided alpha for the bootstrap CI (default 0.05 -> 95% CI).
    seed : int
        Seed for the bootstrap RNG so reruns are byte-identical.

    Returns
    -------
    dict | None
        ``None`` if fewer than 3 paired observations are available
        (Wilcoxon is under-determined). Otherwise a dict with keys:

        - ``reference``: ``ref``
        - ``other``: ``other``
        - ``mean_diff``: float, ``mean(ref - other)`` over paired datasets
        - ``p_raw``: float, two-sided Wilcoxon signed-rank p-value
          (1.0 if the test cannot be computed, mirroring
          :func:`select_significance_pairs`).
        - ``p_holm``: ``None`` -- caller must populate via
          :func:`holm_correct_pairs` once a family is gathered.
        - ``n_pairs``: int, number of paired observations after dropping
          NaNs in either method.
        - ``ci_lo``, ``ci_hi``: float, bootstrap percentile CI bounds for
          ``mean_diff`` at ``ci_alpha``.

    Examples
    --------
    >>> # Single comparison (returns one record with p_raw, p_holm=None):
    >>> rec = compute_pair_effects(long_df, "ASW", "CCVGAE", "VAE")
    >>> # Family of comparisons (apply Holm together):
    >>> family = [
    ...     compute_pair_effects(long_df, "ASW", "CCVGAE", baseline)
    ...     for baseline in baselines
    ... ]
    >>> family = holm_correct_pairs([r for r in family if r is not None])
    """
    sub = long_df.loc[long_df["metric"] == metric, [pair_col, group_col, "value"]]
    if sub.empty:
        return None

    pivot = sub.pivot_table(
        index=pair_col, columns=group_col, values="value", aggfunc="first"
    )
    if ref not in pivot.columns or other not in pivot.columns:
        return None

    paired = pivot[[ref, other]].dropna()
    n_pairs = int(len(paired))
    if n_pairs < 3:
        return None

    ref_arr = paired[ref].to_numpy(dtype=float)
    oth_arr = paired[other].to_numpy(dtype=float)
    diff = ref_arr - oth_arr
    mean_diff = float(np.mean(diff))

    # Wilcoxon raw p-value (Holm comes later, family-level).
    if np.all(diff == 0):
        p_raw = 1.0
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p_raw_val = wilcoxon(
                    ref_arr, oth_arr, alternative="two-sided", zero_method="wilcox"
                )
            p_raw = float(p_raw_val)
        except (ValueError, RuntimeError):
            p_raw = 1.0

    # Paired bootstrap CI for mean_diff (percentile method, fixed seed).
    rng = np.random.default_rng(seed)
    n_bootstrap = 5000
    idx = rng.integers(0, n_pairs, size=(n_bootstrap, n_pairs))
    boot_means = diff[idx].mean(axis=1)
    lo_pct = 100.0 * (ci_alpha / 2.0)
    hi_pct = 100.0 * (1.0 - ci_alpha / 2.0)
    ci_lo = float(np.percentile(boot_means, lo_pct))
    ci_hi = float(np.percentile(boot_means, hi_pct))

    return {
        "reference": ref,
        "other": other,
        "mean_diff": mean_diff,
        "p_raw": p_raw,
        "p_holm": None,
        "n_pairs": n_pairs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def holm_correct_pairs(pairs: list[dict]) -> list[dict]:
    """Apply Holm correction to a family of :func:`compute_pair_effects` records.

    Mutates each input dict's ``p_holm`` field in place using
    :func:`_holm_correct` over the ``p_raw`` values, then returns the same
    list. Records with ``p_raw`` missing are skipped (left with ``p_holm``
    unchanged), but in practice every dict produced by
    :func:`compute_pair_effects` has a numeric ``p_raw``.
    """
    indexed = [(i, rec) for i, rec in enumerate(pairs) if rec.get("p_raw") is not None]
    if not indexed:
        return pairs
    p_raws = [float(rec["p_raw"]) for _, rec in indexed]
    p_holms = _holm_correct(p_raws)
    for (i, rec), p_h in zip(indexed, p_holms):
        rec["p_holm"] = float(p_h)
    return pairs
