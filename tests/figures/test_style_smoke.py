"""Smoke test for scccvgben.figures.style.create_publication_figure.

Renders a 2-metric synthetic figure and asserts:
- Backend is Agg.
- pdf.fonttype is 42 inside the render context.
- Panel labels A and B are present.
- Significance bracket count <= 3 per panel when reference_method given.
- import REA does not appear anywhere in scccvgben.figures.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from scccvgben.figures import (
    NUMERIC_METRICS,
    apply_publication_rcparams,
    dataset_key_from_result_stem,
    create_publication_figure,
    preliminary_path,
    select_significance_pairs,
)


def _synthetic_long_df(n_datasets: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    methods = ["scCCVGBen_GAT", "GCN", "PCA", "scVI"]
    metrics = ["ARI", "NMI"]
    rows = []
    for ds in range(n_datasets):
        for m in methods:
            base = {"scCCVGBen_GAT": 0.85, "GCN": 0.55, "PCA": 0.45, "scVI": 0.65}[m]
            for metric in metrics:
                rows.append(
                    {
                        "dataset_id": f"DS{ds:02d}",
                        "dataset_name": f"DS{ds:02d}",
                        "modality": "scrna",
                        "method": m,
                        "metric": metric,
                        "value": float(np.clip(base + rng.normal(0, 0.05), 0, 1)),
                    }
                )
    return pd.DataFrame(rows)


def test_backend_pinned_to_agg():
    assert matplotlib.get_backend().lower() == "agg"


def test_apply_publication_rcparams_idempotent():
    apply_publication_rcparams()
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    apply_publication_rcparams()
    assert matplotlib.rcParams["pdf.fonttype"] == 42


def test_create_publication_figure_renders_without_reference():
    df = _synthetic_long_df()
    fig, axes = create_publication_figure(
        df, metrics=["ARI", "NMI"], reference_method=None,
    )
    assert len(axes) == 2
    panel_letters = []
    for ax in axes:
        for child in ax.texts:
            if child.get_text() in ("A", "B"):
                panel_letters.append(child.get_text())
    assert "A" in panel_letters and "B" in panel_letters
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    assert buf.tell() > 0


def test_create_publication_figure_with_significance_brackets():
    df = _synthetic_long_df()
    fig, axes = create_publication_figure(
        df, metrics=["ARI", "NMI"], reference_method="scCCVGBen_GAT",
    )
    for ax in axes:
        bracket_lines = [
            ln for ln in ax.lines
            if ln.get_color() == "black"
            and len(ln.get_xdata()) == 4
        ]
        assert len(bracket_lines) <= 3


def test_no_REA_imports_in_figures_package():
    pkg_dir = Path(__file__).resolve().parents[2] / "scccvgben" / "figures"
    offending = []
    for py in pkg_dir.glob("*.py"):
        text = py.read_text()
        if "import REA" in text or "from REA" in text:
            offending.append(py.name)
    assert not offending, f"REA imports found in: {offending}"


def test_preliminary_path_grammar():
    p_full = preliminary_path("figures/fig08_scrna_benchmark", n_obs=100, target=100)
    p_part = preliminary_path("figures/fig08_scrna_benchmark", n_obs=78, target=100)
    assert p_full.name == "fig08_scrna_benchmark.pdf"
    assert p_part.name == "fig08_scrna_benchmark.PRELIMINARY.pdf"


def test_significance_pair_selection_handles_small_n():
    df = _synthetic_long_df(n_datasets=2)
    pairs = select_significance_pairs(
        df, metric="ARI", reference_method="scCCVGBen_GAT",
    )
    assert pairs == []


def test_significance_pair_selection_returns_top_k():
    df = _synthetic_long_df(n_datasets=20)
    pairs = select_significance_pairs(
        df, metric="ARI", reference_method="scCCVGBen_GAT", top_k=3,
    )
    assert len(pairs) <= 3
    for ref, other, p in pairs:
        assert ref == "scCCVGBen_GAT"
        assert 0.0 <= p <= 1.0


def test_numeric_metric_catalog_matches_reported_numeric_fields():
    assert len(NUMERIC_METRICS) == 24
    assert "data_type_intrin" not in NUMERIC_METRICS
    assert "interpretation_intrin" not in NUMERIC_METRICS


def test_dataset_key_normalises_result_filename_stems():
    assert dataset_key_from_result_stem("Can_GSE115571_LPSMmDev_df") == "GSE115571_LPSMmDev"
    assert (
        dataset_key_from_result_stem("ATA_GSM5124061_PDX_390_vehicle_peak_bc_matrix_df")
        == "GSM5124061_PDX_390_vehicle_peak_bc_matrix"
    )
    assert dataset_key_from_result_stem("synthetic_200x100") == "synthetic_200x100"


def test_figure_module_smoke_via_importlib():
    spec = importlib.util.find_spec("scccvgben.figures.style")
    assert spec is not None
