"""Figure-rendering helpers for scCCVGBen.

Public surface:
- create_publication_figure: DataFrame-native multi-panel renderer.
- apply_publication_rcparams: idempotent rcParams + seaborn palette reset.
- melt_reconciled: reconciled per-dataset CSV directory -> long DataFrame.
- preliminary_path: filename grammar (figXX_<scope>.PRELIMINARY.pdf).
- select_significance_pairs: Wilcoxon Holm-corrected pair selection.
"""

from __future__ import annotations

from ._long_form import dataset_key_from_result_stem, filter_to_manifest, melt_reconciled
from ._naming import preliminary_path
from ._significance import select_significance_pairs
from .metrics import (
    METRIC_LABELS,
    NON_NUMERIC_METRICS,
    NUMERIC_METRIC_FAMILIES,
    NUMERIC_METRICS,
    add_method_display,
    available_numeric_metrics,
    short_method_name,
)
from .style import apply_publication_rcparams, create_publication_figure

__all__ = [
    "METRIC_LABELS",
    "NON_NUMERIC_METRICS",
    "NUMERIC_METRIC_FAMILIES",
    "NUMERIC_METRICS",
    "add_method_display",
    "apply_publication_rcparams",
    "available_numeric_metrics",
    "create_publication_figure",
    "dataset_key_from_result_stem",
    "filter_to_manifest",
    "melt_reconciled",
    "preliminary_path",
    "select_significance_pairs",
    "short_method_name",
]
