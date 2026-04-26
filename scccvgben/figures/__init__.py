"""Figure-rendering helpers for scCCVGBen.

Public surface:
- create_publication_figure: DataFrame-native multi-panel renderer.
- create_metric_family_figure: BEN/DRE/LSE grouped metric-family renderer.
- create_metric_grid_figure: fixed 4×5, 20-metric publication renderer.
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
    ALL_NUMERIC_METRICS,
    EXCLUDED_PUBLICATION_METRICS,
    METRIC_LABELS,
    METRIC_FAMILY_ROWS,
    METRIC_FAMILY_TITLES,
    METRIC_PANEL_GRID,
    METRIC_TO_FAMILY,
    NON_NUMERIC_METRICS,
    NUMERIC_METRIC_FAMILIES,
    NUMERIC_METRICS,
    add_method_display,
    available_numeric_metrics,
    metric_coverage_audit,
    short_method_name,
    write_metric_audit,
)
from .fonts import arial_font_path, arial_font_paths, register_arial_with_matplotlib
from .style import (
    apply_publication_rcparams,
    create_metric_family_figure,
    create_metric_grid_figure,
    create_publication_figure,
)

__all__ = [
    "METRIC_LABELS",
    "ALL_NUMERIC_METRICS",
    "EXCLUDED_PUBLICATION_METRICS",
    "METRIC_FAMILY_ROWS",
    "METRIC_FAMILY_TITLES",
    "METRIC_PANEL_GRID",
    "METRIC_TO_FAMILY",
    "NON_NUMERIC_METRICS",
    "NUMERIC_METRIC_FAMILIES",
    "NUMERIC_METRICS",
    "add_method_display",
    "apply_publication_rcparams",
    "arial_font_path",
    "arial_font_paths",
    "available_numeric_metrics",
    "create_metric_family_figure",
    "create_metric_grid_figure",
    "create_publication_figure",
    "dataset_key_from_result_stem",
    "filter_to_manifest",
    "melt_reconciled",
    "preliminary_path",
    "select_significance_pairs",
    "short_method_name",
    "metric_coverage_audit",
    "register_arial_with_matplotlib",
    "write_metric_audit",
]
