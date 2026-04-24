"""sklearn_methods.py — Thin wrappers for classical dimensionality-reduction baselines.

Each function follows the convention:
    run_<method>(X: np.ndarray, n_components: int = 10) -> np.ndarray

Returns the latent matrix of shape (n_cells, n_components).
All methods use random_state=42 for reproducibility; whitening is off.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import (
    PCA,
    KernelPCA,
    FastICA,
    FactorAnalysis,
    NMF,
    TruncatedSVD,
    DictionaryLearning,
)


def run_PCA(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Principal Component Analysis (linear)."""
    model = PCA(n_components=n_components, random_state=42)
    return model.fit_transform(X)


def run_KPCA(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Kernel PCA with RBF kernel."""
    model = KernelPCA(n_components=n_components, kernel="rbf", random_state=42)
    return model.fit_transform(X)


def run_ICA(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Independent Component Analysis (FastICA); whitening off."""
    model = FastICA(
        n_components=n_components,
        whiten=False,
        random_state=42,
        max_iter=500,
    )
    return model.fit_transform(X)


def run_FA(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Factor Analysis."""
    model = FactorAnalysis(n_components=n_components, random_state=42)
    return model.fit_transform(X)


def run_NMF(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Non-negative Matrix Factorization. Clips negative values to 0."""
    X_nn = np.clip(X, 0, None)
    model = NMF(n_components=n_components, random_state=42, max_iter=400)
    return model.fit_transform(X_nn)


def run_TSVD(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Truncated SVD (LSA-style, works on sparse and dense)."""
    model = TruncatedSVD(n_components=n_components, random_state=42)
    return model.fit_transform(X)


def run_DICL(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Dictionary Learning (sparse coding baseline)."""
    model = DictionaryLearning(
        n_components=n_components,
        random_state=42,
        max_iter=200,
        transform_algorithm="lasso_lars",
    )
    return model.fit_transform(X)


# Dispatch map used by runner.py
SKLEARN_REGISTRY: dict[str, callable] = {
    "PCA": run_PCA,
    "KPCA": run_KPCA,
    "ICA": run_ICA,
    "FA": run_FA,
    "NMF": run_NMF,
    "TSVD": run_TSVD,
    "DICL": run_DICL,
}
