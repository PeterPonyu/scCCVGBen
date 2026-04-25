# Ported from the reference benchmark LSE module (revised 2026-04-23)
"""Latent Space Evaluator — trajectory directionality metric.

Public entry point: ``trajectory_directionality(latent_space)``.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA


def trajectory_directionality(latent_space: np.ndarray) -> float:
    """Assess how strongly a single primary developmental axis dominates.

    Ported from ``SingleCellLatentSpaceEvaluator.trajectory_directionality_score``
    in scCCVGBen/LSE.py.  The class method is exposed here as a free function for
    convenience.

    The score measures the dominance of the first principal component relative
    to all remaining components.  A high value (→ 1) indicates a strong
    directional trajectory; a low value (→ 0) indicates isotropic / undirected
    layout.

    Parameters
    ----------
    latent_space : (n_samples, n_dims) array representing the latent embedding.

    Returns
    -------
    float in [0, 1].
    """
    try:
        pca = PCA()
        pca.fit(latent_space)
        explained_var = pca.explained_variance_ratio_

        if len(explained_var) >= 2:
            other_variance = float(np.sum(explained_var[1:]))
            if other_variance > 1e-10:
                dominance_ratio = float(explained_var[0]) / other_variance
                # sigmoid-style normalisation identical to scCCVGBen source
                directionality = dominance_ratio / (1.0 + dominance_ratio)
            else:
                directionality = 1.0
        else:
            directionality = 1.0

        return float(np.clip(directionality, 0.0, 1.0))

    except Exception as exc:
        warnings.warn(f"trajectory_directionality error: {exc}")
        return 0.5
