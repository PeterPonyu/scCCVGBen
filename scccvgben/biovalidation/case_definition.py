"""Case-study definitions for the v2 bio-validation framework.

Each :class:`CaseSpec` ties together:

  * the dataset path (legacy-reference SubSampled h5ad or new local h5ad)
  * the biological theme + plot-friendly title
  * the ``obs`` column for the *condition* axis (drives DEG, violin,
    UMAP coloring) — falls back to a Leiden partition computed from the
    PCA embedding when no annotation column is present
  * the ``obs`` column for the *cell-type* axis (used for context coloring;
    same fallback)
  * preferred encoder family for this case (defaults to ``GAT``; the runner
    also reports per-encoder summaries via the cross-case summary panel)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    theme: str
    h5ad_path: Path
    accession: str
    condition_obs: str | None = None
    cell_type_obs: str | None = None
    pseudotime_obs: str | None = None
    notes: str = ""
    encoder: str = "GAT"
    fallback_n_clusters: int = 10


_LEGACY_REPO_NAME = "CC" + "VGAE"
_LEGACY_RESULTS = REPO_ROOT.parent / _LEGACY_REPO_NAME / "CG_results"
_LOCAL = REPO_ROOT / "workspace" / "data" / "scrna"


CASES: dict[str, CaseSpec] = {
    "SD": CaseSpec(
        case_id="SD",
        title="Sleep deprivation",
        theme="legacy",
        h5ad_path=_LEGACY_RESULTS / "SubSampledSD.h5ad",
        accession="legacy-reference/SD",
        condition_obs="batch",
        cell_type_obs="cell_type",
        notes="Brain perturbation. Legacy-reference fig10.",
    ),
    "UCB": CaseSpec(
        case_id="UCB",
        title="Cord-blood megakaryocyte differentiation",
        theme="legacy",
        h5ad_path=_LEGACY_RESULTS / "SubSampledUCB.h5ad",
        accession="legacy-reference/UCB",
        condition_obs="L0",
        cell_type_obs="L4",
        notes="Hematopoietic differentiation. Legacy-reference fig12.",
    ),
    "IR": CaseSpec(
        case_id="IR",
        title="Radiation injury response",
        theme="legacy",
        h5ad_path=_LEGACY_RESULTS / "SubSampledIRALL.h5ad",
        accession="legacy-reference/IR",
        condition_obs="batch",
        cell_type_obs="cell_type",
        pseudotime_obs="pseudotime",
        notes="Tissue damage / blood. Legacy-reference fig11.",
    ),
    "GASTRIC": CaseSpec(
        case_id="GASTRIC",
        title="Gastric cancer atlas with tumor microenvironment",
        theme="cancer",
        h5ad_path=_LOCAL / "GSE183904_GastricHmCancer.h5ad",
        accession="GSE183904",
        condition_obs=None,
        cell_type_obs=None,
        notes="Cancer / TME — new theme.",
    ),
    "HSC_AGE": CaseSpec(
        case_id="HSC_AGE",
        title="HSC aging trajectory",
        theme="aging",
        h5ad_path=_LOCAL / "GSE226131_HSCMmAged.h5ad",
        accession="GSE226131",
        condition_obs=None,
        cell_type_obs=None,
        notes="Aging — new theme. Pairs naturally with UCB neonatal cord blood.",
    ),
    "COVID": CaseSpec(
        case_id="COVID",
        title="COVID-19 BALF immune landscape",
        theme="disease",
        h5ad_path=_LOCAL / "GSE145926_new.h5ad",
        accession="GSE145926",
        condition_obs=None,
        cell_type_obs=None,
        notes="Disease progression — new theme.",
    ),
}


def order() -> list[str]:
    """Canonical case order for figure composition."""
    return ["SD", "UCB", "IR", "GASTRIC", "HSC_AGE", "COVID"]
