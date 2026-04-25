"""Vendored reference benchmark core modules.

Source-of-truth modules are copied from the external reference benchmark tree:
  - cgvae/     : graph VAE training + model agent
  - lse.py     : SingleCellLatentSpaceEvaluator
  - dre.py     : evaluate_dimensionality_reduction

Purpose: guarantee bit-for-bit compatibility with CG_dl_merged reused results.
Any modification here breaks reuse and should be reflected in the reference tree.

Use as:
    from scccvgben.external.reference_core.cgvae import CGVAE_agent
    from scccvgben.external.reference_core.lse import evaluate_single_cell_latent_space
    from scccvgben.external.reference_core.dre import evaluate_dimensionality_reduction
"""
