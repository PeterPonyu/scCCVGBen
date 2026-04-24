"""Vendored CCVGAE revised-benchmark core modules.

Source-of-truth code copied verbatim from /home/zeyufu/LAB/CCVGAE/:
  - cgvae/     : /home/zeyufu/LAB/CCVGAE/CGVAE/  (training + model agent)
  - lse.py     : /home/zeyufu/LAB/CCVGAE/LSE.py  (SingleCellLatentSpaceEvaluator)
  - dre.py     : /home/zeyufu/LAB/CCVGAE/DRE.py  (evaluate_dimensionality_reduction)

Purpose: guarantee bit-for-bit compatibility with CG_dl_merged reused results.
Any modification here breaks reuse — edits should be upstreamed to CCVGAE.

Use as:
    from scccvgben.external.ccvgae_core.cgvae import CGVAE_agent
    from scccvgben.external.ccvgae_core.lse import evaluate_single_cell_latent_space
    from scccvgben.external.ccvgae_core.dre import evaluate_dimensionality_reduction
"""
