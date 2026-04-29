from __future__ import annotations

import inspect
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts import make_all_figures
from scripts.make_all_figures import FigureSpec
from scccvgben.biovalidation import sidecar
from scccvgben.biovalidation.visualize import gene_grid, scatter
from scccvgben.biovalidation.visualize.case_style import case_cmap


def test_case_colormaps_match_pair_go_palette_contract():
    assert case_cmap("SD") == "Blues"
    assert case_cmap("GASTRIC") == "Reds"
    assert case_cmap("UCB") == "YlOrBr"
    assert case_cmap("HSC_AGE") == "Greens"
    assert case_cmap("IR") == "Purples"
    assert case_cmap("COVID") == "OrRd"


def test_gene_grid_threads_case_cmap_to_expression_umaps(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_render_continuous_scatter(ax, umap, values, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        gene_grid,
        "render_continuous_scatter",
        fake_render_continuous_scatter,
    )

    fig, ax = plt.subplots()
    try:
        gene_grid.render_gene_grid(
            ax,
            umap=np.zeros((3, 2)),
            latent=np.zeros((3, 1)),
            top_k_df=pd.DataFrame(
                [{"dim": 0, "rank": 0, "gene": "GeneA", "rho": 0.42}]
            ),
            expression=pd.DataFrame({"GeneA": [0.1, 0.2, 0.3]}),
            cmap="Reds",
        )
    finally:
        plt.close(fig)

    assert calls
    assert calls[0]["cmap"] == "Reds"


def test_biovalidation_pair_completeness_counts_only_required_cases(tmp_path, monkeypatch):
    bio = tmp_path / "figures" / "biovalidation"
    bio.mkdir(parents=True)
    for case_id in ("SD", "GASTRIC", "UCB", "HSC_AGE", "IR", "COVID"):
        (bio / f"fig_biovalidation_case_{case_id}.pdf").touch()
        (bio / f"fig_biovalidation_case_{case_id}.png").touch()

    monkeypatch.setattr(make_all_figures, "REPO_ROOT", tmp_path)

    fig09 = FigureSpec("fig09", "scripts.make_biovalidation_pairs", "*.pdf", 2, "biovalidation_pair")
    assert make_all_figures._observed_n(fig09) == 2

    (bio / "fig_biovalidation_case_GASTRIC.png").unlink()
    assert make_all_figures._observed_n(fig09) == 1

    fig10 = FigureSpec("fig10", "scripts.make_biovalidation_pairs", "*.pdf", 2, "biovalidation_pair")
    assert make_all_figures._observed_n(fig10) == 2


def test_pair_dotplot_legend_uses_ascii_safe_log10_label():
    from scripts import make_biovalidation_pairs

    source = inspect.getsource(make_biovalidation_pairs._draw_dotplot_key)
    assert "-log10(padj)" in source
    assert "\u2081" not in source
    assert "\u2080" not in source


def test_extended_biovalidation_pair_writes_source_manifest():
    from scripts import make_biovalidation_pairs

    source = inspect.getsource(make_biovalidation_pairs._compose_extended_pair)
    assert "_write_source_manifest" in source
    assert "source_manifest_path" in source


def test_sidecar_caches_latent_corr_matrix():
    payload = {
        "case": SimpleNamespace(
            case_id="T",
            title="test",
            theme="test",
            accession="GSETEST",
            encoder="GAT",
            condition_obs="batch",
            cell_type_obs="cell_type",
        ),
        "latent": np.array([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]]),
        "condition": pd.Series(["a", "a", "b"]),
        "latent_corr": np.array([[1.0, 0.25], [0.25, 1.0]]),
        "top_k_genes_df": pd.DataFrame(),
        "enrichment_df": pd.DataFrame(),
    }

    data = sidecar.build_case_sidecar(payload)

    assert data["version"] == sidecar.SIDECAR_VERSION
    assert data["latent_corr"] == [[1.0, 0.25], [0.25, 1.0]]


def test_pair_latent_corr_panel_uses_cached_rdbu_matrix():
    from PIL import Image
    from scripts import make_biovalidation_pairs

    card = Image.new("RGB", (420, 340), "white")
    ok = make_biovalidation_pairs._paste_latent_corr_panel(
        card,
        xy=(10, 10),
        size=(360, 300),
        case_id="SD",
        sidecar={"latent_corr": [[1.0, 0.2], [0.2, 1.0]]},
        label="D",
        title="Latent self-corr",
    )

    assert ok
    assert card.getbbox() is not None


def test_categorical_scatter_shows_label_list_for_many_categories():
    fig, ax = plt.subplots()
    try:
        scatter.render_categorical_scatter(
            ax,
            np.column_stack([np.arange(12), np.zeros(12)]),
            pd.Series([f"type_{i}" for i in range(12)]),
            legend_loc="right",
            legend_title="cell type",
            max_legend=4,
            summary_limit=5,
        )
        text = "\n".join(t.get_text() for t in ax.texts)
    finally:
        plt.close(fig)

    assert "12 categories" not in text
    assert "type_0" in text
    assert "+7 more" in text


def test_categorical_scatter_compacts_geo_sample_filenames():
    fig, ax = plt.subplots()
    try:
        scatter.render_categorical_scatter(
            ax,
            np.column_stack([np.arange(4), np.zeros(4)]),
            pd.Series([
                "GSM5573480_sample15.csv.gz",
                "GSM5573499_sample34.csv.gz",
                "GSM5573480_sample15.csv.gz",
                "other",
            ]),
            legend_loc="right",
            legend_title="condition",
        )
        text = "\n".join(t.get_text() for t in ax.get_legend().get_texts())
    finally:
        plt.close(fig)

    assert "s15" in text
    assert "s34" in text
    assert "GSM5573480_sample15" not in text
