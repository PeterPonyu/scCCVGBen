"""Compute primitives for the bio-validation framework.

Each function returns a plain ``dict`` / ``numpy.ndarray`` so the visualize
layer never imports from compute directly — the runner orchestrates and
hands payloads in.
"""
from .latent_gene_corr import top_k_genes_per_dim, latent_self_correlation
from .case_run import run_case

__all__ = ["top_k_genes_per_dim", "latent_self_correlation", "run_case"]
