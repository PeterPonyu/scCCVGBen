"""scCCVGBen quickstart — PBMC 3k example.

Loads the 2,700-cell PBMC dataset bundled with scanpy (no separate download),
preprocesses it, constructs a kNN graph, trains scCCVGBen for 5 epochs, computes
a UMAP from the latent representation, and prints the latent shape together with
ARI / NMI against the Leiden ground-truth labels if they are available.

Run:
    python examples/quickstart.py
"""

import numpy as np
import scanpy as sc

# ------------------------------------------------------------------
# 1. Load PBMC 3k (included with scanpy, ~2.7 MB download on first run)
# ------------------------------------------------------------------
adata = sc.datasets.pbmc3k()
print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

# ------------------------------------------------------------------
# 2. Preprocess: QC filter -> log-normalise -> HVG selection -> PCA
# ------------------------------------------------------------------
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable].copy()
adata.layers["counts"] = adata.X.copy()   # CGVAE_agent expects raw in 'counts'
sc.pp.pca(adata, n_comps=50)

# ------------------------------------------------------------------
# 3. Construct a 15-nearest-neighbour kNN graph
# ------------------------------------------------------------------
sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")

# ------------------------------------------------------------------
# 4. Train scCCVGBen for 5 epochs via CGVAE_agent
# ------------------------------------------------------------------
from scccvgben.external.reference_core.cgvae import CGVAE_agent

agent = CGVAE_agent(adata, graph_type="GAT", latent_dim=10, hidden_dim=128,
                    hidden_layers=2, i_dim=5, epochs=5, lr=1e-4,
                    w_recon=1.0, w_irecon=1.0, w_kl=1.0, w_adj=1.0,
                    alpha=0.5, dropout=0.05, subgraph_size=300,
                    num_subgraphs_per_epoch=10, n_neighbors=15, tech="PCA")
agent.fit(epochs=5, silent=False)

latent = agent.get_latent()   # numpy array (n_cells, latent_dim)
print(f"Latent shape: {latent.shape}")

# ------------------------------------------------------------------
# 5. UMAP from the latent representation
# ------------------------------------------------------------------
import anndata as ad
adata_lat = ad.AnnData(X=latent, obs=adata.obs.copy())
sc.pp.neighbors(adata_lat, use_rep="X", n_neighbors=15)
sc.tl.umap(adata_lat)
print("UMAP computed; coordinates in adata_lat.obsm['X_umap']")

# ------------------------------------------------------------------
# 6. ARI / NMI against Leiden labels (or latent shape if absent)
# ------------------------------------------------------------------
sc.tl.leiden(adata_lat, resolution=1.0)
predicted = adata_lat.obs["leiden"].values

# pbmc3k_processed has 'louvain' labels; try both common column names
label_col = next((c for c in ("louvain", "cell_type", "leiden_true")
                  if c in adata.obs.columns), None)
if label_col is not None:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    true_labels = adata.obs[label_col].values
    ari = adjusted_rand_score(true_labels, predicted)
    nmi = normalized_mutual_info_score(true_labels, predicted)
    print(f"ARI={ari:.4f}  NMI={nmi:.4f}  (vs '{label_col}' labels)")
else:
    print("No ground-truth label column found; latent shape:", latent.shape)
