"""Test statistical helpers: Wilcoxon+Holm, Cliff's delta, CV."""

from __future__ import annotations

import numpy as np
import pytest

from scccvgben.stats import wilcoxon_signed_rank_with_holm, cliff_delta, coefficient_of_variation

RNG = np.random.default_rng(7)
METRICS = ["nmi", "ari", "asw", "silhouette"]

# a clearly dominates b
_A = {m: RNG.normal(1.0, 0.1, 30) for m in METRICS}
_B = {m: RNG.normal(0.5, 0.1, 30) for m in METRICS}


def test_wilcoxon_holm_all_significant():
    """All 4 metrics should be significant when a >> b."""
    df = wilcoxon_signed_rank_with_holm(_A, _B, alpha=0.05)
    assert set(df["metric"]) == set(METRICS)
    assert df["significant"].all(), f"Expected all significant:\n{df}"
    assert (df["p_holm"] < 0.05).all()


def test_cliff_delta_large_positive():
    """Cliff's delta should be close to +1 when a >> b."""
    a = RNG.normal(1.0, 0.05, 50)
    b = RNG.normal(0.0, 0.05, 50)
    d = cliff_delta(a, b)
    assert d > 0.8, f"Expected delta > 0.8, got {d:.3f}"


def test_coefficient_of_variation_small():
    """CV of tight distribution should be small."""
    x = RNG.normal(10.0, 0.5, 100)
    cv = coefficient_of_variation(x)
    assert cv < 0.5, f"Expected CV < 0.5, got {cv:.3f}"


def test_wilcoxon_holm_columns():
    """Returned DataFrame has expected columns."""
    df = wilcoxon_signed_rank_with_holm(_A, _B)
    for col in ["metric", "statistic", "p_raw", "p_holm", "significant"]:
        assert col in df.columns, f"Missing column '{col}'"


def test_cliff_delta_symmetry():
    """cliff_delta(a, b) == -cliff_delta(b, a)."""
    a = RNG.normal(1.0, 0.1, 20)
    b = RNG.normal(0.0, 0.1, 20)
    assert abs(cliff_delta(a, b) + cliff_delta(b, a)) < 1e-10
