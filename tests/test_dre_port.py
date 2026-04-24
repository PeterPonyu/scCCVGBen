"""Regression tests for ported DRE / LSE Q-metrics and trajectory directionality."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA

from scccvgben.training.dre import evaluate_dimensionality_reduction
from scccvgben.training.lse import trajectory_directionality

RNG = np.random.default_rng(42)

# ── fixtures ───────────────────────────────────────────────────────────────────

def _synthetic_pair(n: int = 100, d_high: int = 10, d_low: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_high, X_low) with a known structure (PCA of X_high)."""
    X_high = RNG.standard_normal((n, d_high))
    pca = PCA(n_components=d_low, random_state=0)
    X_low = pca.fit_transform(X_high)
    return X_high, X_low


def _trajectory_data(n: int = 100, d: int = 10) -> np.ndarray:
    """Synthetic data with a dominant first PC (strong trajectory)."""
    t = np.linspace(0, 1, n)
    X = RNG.standard_normal((n, d)) * 0.1
    X[:, 0] += t * 5  # dominant axis
    return X


# ── evaluate_dimensionality_reduction ─────────────────────────────────────────

def test_dre_returns_expected_keys():
    """Result dict must contain exactly the five required keys."""
    X_high, X_low = _synthetic_pair()
    result = evaluate_dimensionality_reduction(X_high, X_low, verbose=False)
    expected_keys = {"distance_correlation", "Q_local", "Q_global", "K_max", "overall_quality"}
    assert set(result.keys()) == expected_keys, (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )


def test_dre_all_finite():
    """Every value in the result dict must be finite."""
    X_high, X_low = _synthetic_pair()
    result = evaluate_dimensionality_reduction(X_high, X_low, verbose=False)
    for key, val in result.items():
        assert np.isfinite(val), f"result['{key}'] = {val} is not finite"


def test_dre_distance_correlation_in_range():
    """Spearman distance correlation should be in [-1, 1]."""
    X_high, X_low = _synthetic_pair()
    result = evaluate_dimensionality_reduction(X_high, X_low, verbose=False)
    assert -1.0 <= result["distance_correlation"] <= 1.0


def test_dre_q_values_in_range():
    """Q_local, Q_global, overall_quality must be in [0, 1]."""
    X_high, X_low = _synthetic_pair()
    result = evaluate_dimensionality_reduction(X_high, X_low, verbose=False)
    for key in ("Q_local", "Q_global", "overall_quality"):
        assert 0.0 <= result[key] <= 1.0, f"result['{key}'] = {result[key]} out of [0,1]"


def test_dre_k_max_positive():
    """K_max must be a positive integer."""
    X_high, X_low = _synthetic_pair()
    result = evaluate_dimensionality_reduction(X_high, X_low, verbose=False)
    assert isinstance(result["K_max"], (int, np.integer))
    assert result["K_max"] >= 1


def test_dre_pca_embedding_has_higher_correlation():
    """PCA embedding should have higher distance_corr than a random embedding."""
    X_high, X_low_pca = _synthetic_pair(n=80, d_high=10, d_low=2)
    X_low_rand = RNG.standard_normal((80, 2))
    r_pca = evaluate_dimensionality_reduction(X_high, X_low_pca, verbose=False)["distance_correlation"]
    r_rand = evaluate_dimensionality_reduction(X_high, X_low_rand, verbose=False)["distance_correlation"]
    assert r_pca > r_rand, (
        f"Expected PCA ({r_pca:.3f}) > random ({r_rand:.3f}) distance correlation"
    )


# ── trajectory_directionality ─────────────────────────────────────────────────

def test_trajectory_directionality_in_range():
    """Score must lie in [0, 1] for trajectory-structured data."""
    X = _trajectory_data()
    score = trajectory_directionality(X)
    assert 0.0 <= score <= 1.0, f"trajectory_directionality = {score} not in [0, 1]"


def test_trajectory_directionality_high_for_strong_axis():
    """A strongly directional dataset should score > 0.5."""
    X = _trajectory_data(n=100, d=10)
    score = trajectory_directionality(X)
    assert score > 0.5, f"Expected score > 0.5 for strong trajectory, got {score:.4f}"


def test_trajectory_directionality_low_for_isotropic():
    """Isotropic (spherical) data should score lower than directional data."""
    X_isotropic = RNG.standard_normal((100, 10))
    X_directional = _trajectory_data(n=100, d=10)
    score_iso = trajectory_directionality(X_isotropic)
    score_dir = trajectory_directionality(X_directional)
    assert score_dir > score_iso, (
        f"Directional ({score_dir:.4f}) should exceed isotropic ({score_iso:.4f})"
    )


def test_trajectory_directionality_1d_input():
    """Single-column input should not raise and should return 1.0."""
    X = RNG.standard_normal((50, 1))
    score = trajectory_directionality(X)
    assert score == 1.0
