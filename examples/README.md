# scCCVGBen — Quickstart Example

This directory contains a minimal, self-contained example that demonstrates the
full scCCVGBen pipeline on the publicly available PBMC 3k dataset bundled with
`scanpy` (no separate download required beyond the first `scanpy` cache).

## Requirements

```
scanpy
torch
torch_geometric
anndata
scikit-learn
umap-learn
```

Install the package itself from the repository root:

```bash
pip install -e .
```

## Running

```bash
python examples/quickstart.py
```

Expected runtime on CPU: under 5 minutes for 5 training epochs.

## What the script does

1. Loads `scanpy.datasets.pbmc3k()` — 2,700 PBMC cells, 32,738 genes.
2. Preprocesses: QC filter, log-normalisation, HVG selection (2,000 genes), PCA (50 PCs).
3. Constructs a 15-nearest-neighbour kNN graph via `scanpy.pp.neighbors`.
4. Trains `CGVAE_agent` (GAT encoder, latent dim 10) for 5 epochs.
5. Computes a UMAP from the 10-dimensional latent embedding.
6. Prints ARI / NMI against Leiden cluster labels, or the latent shape if no
   ground-truth annotations are present.
