"""scccvgben.biovalidation — multi-case biological-validation framework v2.

Replaces the original reference bio-validation (3 case studies, 10 latent×gene
pair UMAPs each) with a 6-case multi-panel pipeline that adds:

  * latent-space self-correlation diagnostics
  * top-K (rather than top-1) gene-per-dim correlations
  * pseudotime trajectory inference (PAGA / dpt)
  * condition-stratified DEG and per-dim violins
  * GO / KEGG enrichment per latent dim
  * cross-case quantitative summary panel

Six standardized cases:

  1. SD       — sleep-deprivation perturbation        (legacy reference)
  2. UCB      — cord-blood megakaryocyte differentiation (legacy reference)
  3. IR       — radiation-injury response             (legacy reference)
  4. GASTRIC  — gastric cancer + tumor microenvironment (NEW: GSE183904)
  5. HSC_AGE  — hematopoietic stem cell aging         (NEW: GSE226131)
  6. COVID    — COVID-19 BALF immune landscape        (NEW: GSE145926)

Layout / panel sizes are standardized via :class:`Panel` so the figure
composer can stitch panels without aspect-ratio or DPI drift.
"""
from __future__ import annotations

from .case_definition import CASES, CaseSpec

__all__ = ["CASES", "CaseSpec"]
