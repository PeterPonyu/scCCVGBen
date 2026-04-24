"""Shared pytest fixtures for scCCVGBen tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_h5ad_dir(tmp_path_factory):
    """Create a tmp dir with one synthetic h5ad (200 cells x 100 genes, Poisson counts)."""
    import anndata

    tmp = tmp_path_factory.mktemp("h5ad")
    rng = np.random.default_rng(42)

    n_cells, n_genes = 200, 100
    X = rng.poisson(lam=2.0, size=(n_cells, n_genes)).astype(np.float32)

    labels = rng.choice(["A", "B", "C", "D"], size=n_cells).tolist()
    obs = {"cell_type": labels}

    adata = anndata.AnnData(X=X)
    adata.obs["cell_type"] = labels

    out = tmp / "synthetic_200x100.h5ad"
    adata.write_h5ad(str(out))
    return tmp
