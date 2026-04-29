#!/usr/bin/env python3
"""make_biovalidation_pairs.py — pair bio-validation cases into main + detail figures.

The manuscript keeps three bio-validation figures by pairing two biological
case studies per figure.  The main figures are concise narrative cards: each
case shows condition structure, the first retained latent, the condition ×
latent summary, and the strongest GO biological-process evidence.  Dense
supporting evidence (cell-state map, latent self-correlation, top-gene table,
latent-gene mini-UMAP grid, and the fuller GO dot plot) is emitted alongside
each main figure as an extended detail output with its own traceability
manifest.

The compositor intentionally does **not** run model training or result
recomputation in its default path: it uses the cached per-case PNG/PDF
artifacts under ``figures/biovalidation`` and sidecar JSON payloads.

Pairs (default):

    fig09 — SD + GASTRIC      (sleep deprivation + gastric cancer)
    fig10 — UCB + HSC_AGE     (cord-blood megakaryocyte + aged HSC)
    fig11 — IR + COVID        (radiation injury + COVID-19 BALF)

Main figures use a global label sequence A-D for the left case and E-H for the
right case.  Extended detail outputs keep the full single-case panel family
with A-H for the left case card and I-P for the right card.  The original
single-case panel roles are stored in manifests rather than repeated as visible
letters.

Inputs
------
The script re-uses the per-case PNG/PDF artifacts produced by
``run_biovalidation``. Missing inputs are treated as an error unless
``--allow-recompute`` is explicitly supplied.

Source contract
---------------
The durable deliverables are this script, the JSON sidecar/manifests, and the
LaTeX manuscript source.  Rendered PNG/PDF files are regenerated local
review/QA evidence and should not be treated as independent source artifacts.

Usage::

    python scripts/make_biovalidation_pairs.py                 # all 3 pairs
    python scripts/make_biovalidation_pairs.py --pairs fig09   # one pair only
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=False)
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scccvgben.biovalidation.sidecar import (  # noqa: E402
    LABEL_MAP_FILENAME,
    SIDECAR_VERSION,
    load_case_label_evidence,
    save_case_sidecar as save_payload_sidecar,
)
from scccvgben.biovalidation.visualize.case_style import (  # noqa: E402
    case_accent as _case_accent,
    case_cmap as _case_cmap,
)
from scccvgben.figures.fonts import arial_font_path  # noqa: E402

log = logging.getLogger(__name__)


# Pair definition: out_stem -> (left_case_id, right_case_id, pair_label, output_stem)
_PAIRS: dict[str, tuple[str, str, str, str]] = {
    "fig09": ("SD",  "GASTRIC",
              "Sleep deprivation × Gastric cancer microenvironment",
              "fig09_biovalidation_sd_gastric"),
    "fig10": ("UCB", "HSC_AGE",
              "Cord-blood megakaryocyte × Aged hematopoietic stem cells",
              "fig10_biovalidation_ucb_hsc_age"),
    "fig11": ("IR",  "COVID",
              "Radiation injury × COVID-19 BALF immune landscape",
              "fig11_biovalidation_ir_covid"),
}


CARD_W = 1660
CARD_H = 2080
MAIN_CARD_H = 1140
CANVAS_MARGIN = 24
CARD_GAP = 28
TITLE_H = 0
OUTPUT_DPI = 200
SOURCE_MANIFEST_VERSION = 1
MANUSCRIPT_SOURCE = REPO_ROOT / "manuscript" / "scccvgben" / "sn-article.tex"


def _font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the local Arial family with robust fallbacks."""
    preferred = arial_font_path(
        bold="Bold" in name,
        italic="Italic" in name,
    )
    candidates = [
        preferred,
        REPO_ROOT / "site" / "public" / "fonts" / name,
        REPO_ROOT / "site" / "static" / "fonts" / name,
        REPO_ROOT / "site" / "static" / "fonts" / f"{name}.broken_archive",
    ]
    for path in candidates:
        if path and path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


FONT = _font("Arial.ttf", 30)
FONT_SM = _font("Arial.ttf", 24)
FONT_SUBTITLE = _font("Arial.ttf", 28)
FONT_XS = _font("Arial.ttf", 20)
FONT_BOLD = _font("Arial-Bold.ttf", 34)
FONT_BOLD_SM = _font("Arial-Bold.ttf", 25)
FONT_BOLD_XS = _font("Arial-Bold.ttf", 22)
FONT_BOLD_XXS = _font("Arial-Bold.ttf", 20)
FONT_TITLE = _font("Arial-Bold.ttf", 43)
FONT_PANEL = _font("Arial-Bold.ttf", 50)
FONT_GENE = _font("Arial-Bold.ttf", 27)
FONT_GENE_SM = _font("Arial-Bold.ttf", 23)
FONT_TABLE_DIM = _font("Arial-Bold.ttf", 18)
FONT_TABLE_GENE = _font("Arial-Bold.ttf", 24)
FONT_TABLE_RHO = _font("Arial.ttf", 20)
FONT_GO = _font("Arial.ttf", 14)
# Larger Arial used inside _paste_go_summary_panel for GO terms, z0-z9 ticks,
# and the size/colour legend.  Kept distinct from FONT_GO so the condition ×
# latent summary panel (which packs ±x.x cells at 14 px) is not affected.
FONT_GO_DOTPLOT = _font("Arial.ttf", 21)


def _latent_label(dim: int | str) -> str:
    """Display latent-coordinate labels without colliding with day labels."""
    return f"z{int(dim)}"


def _normalise_latent_label_text(text: str) -> str:
    """Convert legacy d-prefixed latent display labels into z-prefixed labels."""
    return re.sub(r"\bd(\d+)(?=[·:\-\s])", r"z\1", text).replace("ρ", "r")


PANEL_ROLE_ORDER: list[str] = [
    "condition_umap",
    "cell_type_umap",
    "latent_d0_umap",
    "latent_corr",
    "latent_gene_grid",
    "condition_latent_summary",
    "top_gene_table",
    "go_enrichment",
]

MAIN_PANEL_ROLE_ORDER: list[str] = [
    "condition_umap",
    "latent_d0_umap",
    "condition_latent_summary",
    "go_enrichment",
]

SUPPLEMENT_DETAIL_ROLES: list[str] = [
    role for role in PANEL_ROLE_ORDER if role not in MAIN_PANEL_ROLE_ORDER
]

SUPPLEMENT_DETAIL_ITEMS: list[str] = [
    *SUPPLEMENT_DETAIL_ROLES,
    "go_enrichment_full_terms",
]

SOURCE_ROLE_LETTERS: dict[str, str] = {
    "condition_umap": "A",
    "cell_type_umap": "B",
    "latent_d0_umap": "C",
    "latent_corr": "D",
    "latent_gene_grid": "F",
    "condition_latent_summary": "G",
    "top_gene_table": "E",
    "go_enrichment": "H",
}


_F_GENE_LABELS_FALLBACK: dict[str, list[list[str]]] = {
    "SD": [
        ["z0·Gria3 ρ+0.61", "z0·Rnf220 ρ+0.55", "z0·Bst2 ρ+0.48",
         "z0·Cdk6 ρ+0.47", "z0·Arhgap32 ρ+0.47"],
        ["z1·Pde4d ρ+0.67", "z1·Cdk6 ρ+0.65", "z1·Ms4a3 ρ+0.64",
         "z1·Mki67 ρ+0.60", "z1·Top2a ρ+0.59"],
        ["z2·S100a6 ρ-0.69", "z2·Aff3 ρ+0.66", "z2·Lyz2 ρ-0.65",
         "z2·Mmp9 ρ-0.64", "z2·Msi2 ρ+0.61"],
    ],
    "GASTRIC": [
        ["z0·TIMP1 ρ-0.62", "z0·SPARC ρ-0.56", "z0·SERPING1 ρ-0.56",
         "z0·PMP22 ρ-0.54", "z0·CALD1 ρ-0.53"],
        ["z1·CCL4 ρ+0.60", "z1·GZMA ρ+0.50", "z1·KRT19 ρ-0.47",
         "z1·KRT8 ρ-0.47", "z1·GZMB ρ+0.47"],
        ["z2·CTSL ρ-0.54", "z2·A2M ρ-0.54", "z2·SPARC ρ-0.53",
         "z2·IGFBP7 ρ-0.51", "z2·CALD1 ρ-0.51"],
    ],
    "UCB": [
        ["z0·LTBP1 ρ+0.81", "z0·RAB27B ρ+0.81", "z0·LAT ρ+0.79",
         "z0·PROS1 ρ+0.79", "z0·ARHGAP6 ρ+0.76"],
        ["z1·CDK6 ρ-0.82", "z1·ITGA4 ρ-0.82", "z1·ZNF521 ρ-0.80",
         "z1·GP9 ρ+0.77", "z1·ITGB3 ρ+0.76"],
        ["z2·DNM3 ρ+0.78", "z2·ARHGAP6 ρ+0.78", "z2·VCL ρ+0.75",
         "z2·ARHGAP18 ρ+0.75", "z2·ITGB3 ρ+0.74"],
    ],
    "HSC_AGE": [
        ["z0·Trem3 ρ+0.44", "z0·Ms4a3 ρ+0.44", "z0·Ly6c2 ρ+0.43",
         "z0·Hp ρ+0.42", "z0·Anxa3 ρ+0.42"],
        ["z1·Plac8 ρ+0.46", "z1·Ctsg ρ+0.45", "z1·Prtn3 ρ+0.43",
         "z1·Mpo ρ+0.42", "z1·Lgals3 ρ-0.40"],
        ["z2·Plac8 ρ+0.61", "z2·Ctsg ρ+0.56", "z2·Prtn3 ρ+0.55",
         "z2·Mpo ρ+0.52", "z2·Mif ρ+0.42"],
    ],
    "IR": [
        ["z0·Emb ρ-0.70", "z0·Ccl3 ρ-0.66", "z0·Gm15915 ρ+0.63",
         "z0·Ccl4 ρ-0.63", "z0·Satb1 ρ-0.63"],
        ["z1·Anxa3 ρ-0.66", "z1·Hp ρ-0.65", "z1·Trem3 ρ-0.64",
         "z1·Slpi ρ-0.63", "z1·Igsf6 ρ-0.61"],
        ["z2·Lsp1 ρ+0.64", "z2·Rgs1 ρ+0.62", "z2·Trem3 ρ-0.62",
         "z2·Nr4a1 ρ+0.61", "z2·Hp ρ-0.61"],
    ],
    "COVID": [
        ["z0·CD3E ρ-0.64", "z0·IL32 ρ-0.63", "z0·CD2 ρ-0.63",
         "z0·PTPRCAP ρ-0.61", "z0·GIMAP7 ρ-0.59"],
        ["z1·IFI30 ρ+0.78", "z1·SNX10 ρ+0.77", "z1·CTSB ρ+0.76",
         "z1·MAFB ρ+0.75", "z1·PLAUR ρ+0.75"],
        ["z2·CD3E ρ+0.63", "z2·PTPRCAP ρ+0.63", "z2·CD2 ρ+0.63",
         "z2·CD3D ρ+0.58", "z2·IL32 ρ+0.58"],
    ],
}


_H_TERM_LABELS: dict[str, list[str]] = {
    "SD": [
        "defense response",
        "positive regulation of immune system process",
        "process involved in interspecies interaction between organisms",
        "myeloid leukocyte migration",
        "positive regulation of immune response",
        "regulation of immune system process",
        "leukocyte migration",
        "cell activation",
        "innate immune response",
        "lymphocyte activation",
        "taxis",
        "leukocyte chemotaxis",
        "regulation of immune response",
        "defense response to other organism",
        "inflammatory response",
        "cell chemotaxis",
        "positive regulation of response to external stimulus",
        "response to bacterium",
        "granulocyte migration",
        "tube development",
    ],
    "GASTRIC": [
        "defense response",
        "external encapsulating structure organization",
        "regulation of immune system process",
        "collagen fibril organization",
        "process involved in interspecies interaction between organisms",
        "inflammatory response",
        "vasculature development",
        "cell adhesion",
        "defense response to other organism",
        "humoral immune response",
        "collagen metabolic process",
        "positive regulation of immune system process",
        "innate immune response",
        "blood vessel morphogenesis",
        "circulatory system development",
        "tube development",
        "structure formation involved in morphogenesis",
        "exogenous peptide antigen presentation",
        "peptide / polysaccharide antigen presentation",
        "response to wounding",
    ],
    "UCB": [
        "mitotic cell cycle process",
        "mitotic cell cycle",
        "cell cycle process",
        "cell division",
        "cell cycle",
        "organelle fission",
        "mitotic nuclear division",
        "chromosome segregation",
        "nuclear chromosome segregation",
        "mitotic sister chromatid segregation",
        "sister chromatid segregation",
        "hemostasis",
        "platelet activation",
        "regulation of body fluid levels",
        "wound healing",
        "response to wounding",
        "regulation of platelet activation",
        "cell activation",
        "platelet aggregation",
        "homotypic cell cell adhesion",
    ],
    "HSC_AGE": [
        "chromosome segregation",
        "nuclear chromosome segregation",
        "mitotic cell cycle",
        "cell division",
        "cell cycle",
        "mitotic nuclear division",
        "sister chromatid segregation",
        "cell cycle process",
        "organelle fission",
        "mitotic sister chromatid segregation",
        "myeloid leukocyte mediated immunity",
        "immune effector process",
        "defense response",
        "defense response to other organism",
        "process involved in interspecies interaction between organisms",
        "leukocyte mediated immunity",
        "phagocytosis",
        "response to bacterium",
        "defense response to bacterium",
        "myeloid leukocyte migration",
    ],
    "IR": [
        "hemopoiesis",
        "regulation of immune system process",
        "leukocyte differentiation",
        "myeloid cell differentiation",
        "positive regulation of immune system process",
        "myeloid leukocyte differentiation",
        "myeloid leukocyte activation",
        "defense response",
        "myeloid cell homeostasis",
        "regulation of hemopoiesis",
        "defense response to other organism",
        "homeostasis of number of cells",
        "erythrocyte homeostasis",
        "multicellular organismal level homeostasis",
        "antigen processing and presentation",
        "mitotic cell cycle",
        "homeostatic process",
        "regulation of cell population proliferation",
        "humoral immune response",
        "regulation of myoblast differentiation",
    ],
    "COVID": [
        "lymphocyte activation",
        "T cell activation",
        "cell activation",
        "regulation of immune system process",
        "defense response",
        "cell killing",
        "inflammatory response",
        "regulation of immune response",
        "T cell differentiation",
        "adaptive immune response",
        "immune effector process",
        "lymphocyte mediated immunity",
        "leukocyte mediated cytotoxicity",
        "positive regulation of immune system process",
        "process involved in interspecies interaction between organisms",
        "innate immune response",
        "regulation of multicellular organismal process",
        "response to bacterium",
        "defense response to other organism",
        "apoptotic process",
    ],
}


def _ensure_case_artifacts(
    case_id: str,
    biovalidation_dir: Path,
    *,
    allow_recompute: bool = False,
) -> Path:
    """Locate cached case artifacts; optionally re-render only when requested."""
    pdf = biovalidation_dir / f"fig_biovalidation_case_{case_id}.pdf"
    png = biovalidation_dir / f"fig_biovalidation_case_{case_id}.png"
    if pdf.exists() and png.exists():
        return pdf

    if not allow_recompute:
        raise SystemExit(
            f"missing cached case artifact(s) for {case_id}: "
            f"{png.name} / {pdf.name}. Re-run the compute pipeline separately "
            "or pass --allow-recompute explicitly."
        )

    log.info("[%s] cached PDF missing — re-rendering via run_biovalidation", case_id)
    from scccvgben.biovalidation import CASES
    from scccvgben.biovalidation.compute import run_case
    from scccvgben.biovalidation.compose import compose_case_figure
    if case_id not in CASES:
        raise SystemExit(f"unknown case id {case_id!r}")
    payload = run_case(CASES[case_id], epochs=100, top_k_genes=5,
                       enrichment_top_n_genes=100, enrichment_top_terms=8,
                       silent=True)
    compose_case_figure(payload, biovalidation_dir)
    _save_case_sidecar(payload, biovalidation_dir)
    return pdf


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return round(box[2] - box[0]), round(box[3] - box[1])


def _fit_crop(
    src: Image.Image,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
    *,
    bg: str = "white",
) -> Image.Image:
    """Crop ``src`` and fit it inside ``size`` without distortion."""
    panel = _trim_near_white(src.crop(crop))
    panel.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, bg)
    out.paste(panel, ((size[0] - panel.width) // 2, (size[1] - panel.height) // 2))
    return out


def _trim_near_white(panel: Image.Image, *, tolerance: int = 248, margin: int = 12) -> Image.Image:
    """Remove empty white margins from a cached panel crop.

    The per-case renderer leaves large canvas-space gutters around many
    subplots.  Trimming before the case-card resize improves effective font
    and marker size without changing the underlying biological result.
    """
    gray = panel.convert("L")
    mask = gray.point(lambda px: 255 if px < tolerance else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return panel
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(panel.width, x1 + margin)
    y1 = min(panel.height, y1 + margin)
    if x1 <= x0 or y1 <= y0:
        return panel
    return panel.crop((x0, y0, x1, y1))


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: str = "#0F172A",
) -> None:
    # Big bold black panel letter (no surrounding card / pill); keep panel
    # labels neutral while dataset-specific colours live inside the data marks.
    x, y = xy
    draw.text((x, y), text[:1], fill="#0F172A", font=FONT_PANEL)


def _draw_panel_title(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    max_width: int,
    fill: str = "#111827",
) -> None:
    """Draw a panel title without letting narrow cards force wasted width."""
    x, y = xy
    for font in (FONT_BOLD_SM, FONT_BOLD_XS, FONT_BOLD_XXS):
        if _text_size(draw, text, font)[0] <= max_width:
            draw.text((x, y), text, fill=fill, font=font)
            return

    rendered = text
    while len(rendered) > 8 and _text_size(draw, rendered + "…", FONT_BOLD_XXS)[0] > max_width:
        rendered = rendered[:-1]
    draw.text((x, y), rendered.rstrip(" ,;-") + "…", fill=fill, font=FONT_BOLD_XXS)


def _paste_panel(
    card: Image.Image,
    src: Image.Image,
    *,
    xy: tuple[int, int],
    size: tuple[int, int],
    crop: tuple[int, int, int, int],
    label: str,
    title: str,
    accent: str = "#0F172A",
) -> None:
    """Paste one cropped biological panel with a clean label/title strip."""
    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(card)
    _draw_label(draw, (x + 6, y + 2), label, fill=accent)
    _draw_panel_title(draw, (x + 70, y + 14), title, max_width=w - 86)
    image_box = (x + 12, y + 48, w - 24, h - 60)
    fitted = _fit_crop(src, crop, (image_box[2], image_box[3]))
    card.paste(fitted, (image_box[0], image_box[1]))


def _format_cluster_key_line(record: dict[str, Any]) -> tuple[str, str]:
    """Return compact key text plus colour for one cluster-label record."""
    cluster_id = str(record.get("cluster_id", "?"))
    label = str(record.get("short_label") or record.get("full_label") or "putative label")
    confidence = str(record.get("confidence", "")).lower()
    suffix = " ?" if confidence in {"low", "putative"} else ""
    fill = "#64748B" if confidence == "low" else "#0F172A"
    return f"{cluster_id} {label}{suffix}", fill


def _paste_cell_type_panel(
    card: Image.Image,
    src: Image.Image,
    *,
    xy: tuple[int, int],
    size: tuple[int, int],
    crop: tuple[int, int, int, int],
    case_id: str,
    biovalidation_dir: Path,
    sidecar: dict[str, Any] | None,
    label: str,
    title: str,
    accent: str = "#0F172A",
) -> bool:
    """Paste the cell-state UMAP with a visible marker-derived cluster key."""
    records = _case_label_records(case_id, biovalidation_dir, sidecar)
    if not records:
        return False

    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(card)
    _draw_label(draw, (x + 6, y + 2), label, fill=accent)
    _draw_panel_title(draw, (x + 70, y + 14), title, max_width=w - 86)

    image_x = x + 10
    image_y = y + 50
    image_h = h - 62
    image_w = max(120, round(w * 0.43))
    fitted = _fit_crop(src, crop, (image_w, image_h))
    card.paste(fitted, (image_x, image_y))

    key_x = image_x + image_w + 10
    key_y = image_y + 2
    key_w = x + w - key_x - 8
    key_h = image_h - 2
    if key_w < 80:
        return True

    draw.text((key_x, key_y), "cluster → marker label", fill="#334155", font=FONT_GO)
    rows = sorted(records, key=_cluster_sort_key)
    n_cols = 2 if len(rows) > 10 and key_w >= 150 else 1
    rows_per_col = math.ceil(len(rows) / n_cols)
    usable_h = max(1, key_h - 24)
    row_h = max(15, min(24, usable_h // max(1, rows_per_col)))
    col_w = key_w // n_cols
    for idx, record in enumerate(rows):
        col = idx // rows_per_col
        row = idx % rows_per_col
        line, fill = _format_cluster_key_line(record)
        tx = key_x + col * col_w
        ty = key_y + 24 + row * row_h
        _draw_fit_left(draw, (tx, ty), col_w - 4, line, fill=fill)

    if any(str(record.get("confidence", "")).lower() == "low" for record in rows):
        note = "? low confidence"
        nw, _ = _text_size(draw, note, FONT_GO)
        draw.text((key_x + max(0, key_w - nw), y + h - 17), note, fill="#64748B", font=FONT_GO)
    return True


def _draw_fit_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    width: int,
    text: str,
    *,
    fill: str = "#0F172A",
) -> None:
    """Draw one mini-panel label, shrinking only when it would collide."""
    x, y = xy
    font = FONT_GENE
    tw, th = _text_size(draw, text, font)
    if tw > width - 8:
        font = FONT_GENE_SM
        tw, th = _text_size(draw, text, font)
    draw.text((x + max(0, (width - tw) // 2), y), text, fill=fill, font=font)


def _case_pdf_text(case_id: str, biovalidation_dir: Path) -> str:
    """Extract vector text from the cached case PDF when poppler is available."""
    pdf_path = biovalidation_dir / f"fig_biovalidation_case_{case_id}.pdf"
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return proc.stdout


def _format_gene_label(dim: int, gene: str, rho: float) -> str:
    return f"{_latent_label(dim)}·{gene} r{rho:+.2f}"


def _gene_grid_labels_from_records(records: list[dict[str, Any]]) -> list[list[str]]:
    """Build the 3×5 mini-UMAP labels from sidecar top-k rows."""
    if not records:
        return []
    out: list[list[str]] = []
    for dim in range(3):
        dim_rows = []
        for row in records:
            try:
                if int(row.get("dim", -1)) == dim:
                    dim_rows.append(row)
            except (TypeError, ValueError):
                continue
        dim_rows = sorted(dim_rows, key=lambda r: int(r.get("rank", 999)))[:5]
        if len(dim_rows) < 5:
            return []
        out.append([
            _format_gene_label(dim, str(r.get("gene", "")), float(r.get("rho", 0.0)))
            for r in dim_rows
        ])
    return out


def _extract_f_gene_labels(
    case_id: str,
    biovalidation_dir: Path,
    sidecar: dict[str, Any] | None = None,
) -> list[list[str]]:
    """Read gene-grid labels from sidecar data, then cached PDF, then fallback text."""
    labels = _gene_grid_labels_from_records((sidecar or {}).get("top_k_genes_df") or [])
    if labels:
        return labels
    text = _case_pdf_text(case_id, biovalidation_dir)
    if text:
        pdf_labels = re.findall(r"[dz]\d+·[^\s]+\s+[ρr][+-]\d+\.\d+", text)
        if len(pdf_labels) >= 15:
            pdf_label_rows = [_normalise_latent_label_text(label) for label in pdf_labels]
            return [pdf_label_rows[i:i + 5] for i in range(0, 15, 5)]
    return [
        [_normalise_latent_label_text(label) for label in row]
        for row in _F_GENE_LABELS_FALLBACK.get(case_id, [])
    ]


def _coerce_top_gene_rows(records: list[dict[str, Any]]) -> list[tuple[int, str, float]]:
    rows: list[tuple[int, str, float]] = []
    for row in records:
        try:
            rows.append((int(row["dim"]), str(row["gene"]), float(row["rho"])))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(rows, key=lambda x: x[0])


def _extract_top_gene_rows(
    case_id: str,
    biovalidation_dir: Path,
    sidecar: dict[str, Any] | None = None,
) -> list[tuple[int, str, float]]:
    """Read top-gene rows from sidecar data, then cached PDF text.

    Sidecars generated by ``run_biovalidation`` are preferred.  PDF text
    extraction remains only as a compatibility bridge for existing cached
    cases that were computed before the sidecar contract was expanded.
    """
    sidecar_rows = _coerce_top_gene_rows((sidecar or {}).get("top_gene_rows") or [])
    if sidecar_rows:
        return sidecar_rows
    text = _case_pdf_text(case_id, biovalidation_dir)
    row_map: dict[int, tuple[int, str, float]] = {}
    if text:
        line_re = re.compile(
            r"^\s*(\d+)\s+([A-Za-z0-9_.:-]+)\s+([+-]\d+\.\d+)"
            r"(?:\s+(\d+)\s+([A-Za-z0-9_.:-]+)\s+([+-]\d+\.\d+))?\s*$",
            flags=re.MULTILINE,
        )
        for m in line_re.finditer(text):
            dim = int(m.group(1))
            row_map[dim] = (dim, m.group(2), float(m.group(3)))
            if m.group(4) is not None:
                dim2 = int(m.group(4))
                row_map[dim2] = (dim2, m.group(5), float(m.group(6)))
    return [row_map[d] for d in sorted(row_map) if 0 <= d <= 99]


def _case_sidecar_path(case_id: str, biovalidation_dir: Path) -> Path:
    return biovalidation_dir / f"fig_biovalidation_case_{case_id}.sidecar.json"


def _label_map_path(biovalidation_dir: Path) -> Path:
    return biovalidation_dir / LABEL_MAP_FILENAME


def _label_map_source(biovalidation_dir: Path) -> str | None:
    path = _label_map_path(biovalidation_dir)
    return _rel(path) if path.exists() else None


def _cluster_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    cluster_id = str(record.get("cluster_id", ""))
    try:
        return (int(cluster_id), cluster_id)
    except ValueError:
        return (10**9, cluster_id)


def _case_label_evidence(case_id: str, biovalidation_dir: Path) -> dict[str, Any]:
    """Return sidecar-ready label evidence derived from the committed label map."""
    return load_case_label_evidence(case_id, biovalidation_dir)


def _case_label_records(
    case_id: str,
    biovalidation_dir: Path,
    sidecar: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return sorted marker-derived cluster-label records for one case."""
    records = (sidecar or {}).get("cluster_label_evidence") or []
    if not records:
        records = _case_label_evidence(case_id, biovalidation_dir).get("cluster_label_evidence") or []
    return sorted(records, key=_cluster_sort_key)


def _case_label_manifest_fields(
    case_id: str,
    biovalidation_dir: Path,
    sidecar: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact label provenance fields for panel/source manifests."""
    evidence = dict(_case_label_evidence(case_id, biovalidation_dir))
    evidence.update({
        key: value for key, value in (sidecar or {}).items()
        if key in {
            "cell_type_label_source",
            "cell_type_label_map",
            "cell_type_label_map_schema",
            "cell_type_label_cluster_key",
        }
    })
    records = _case_label_records(case_id, biovalidation_dir, sidecar)
    if not evidence and not records:
        return {}
    fields = {
        "cell_type_label_source": evidence.get("cell_type_label_source"),
        "cell_type_label_map": evidence.get("cell_type_label_map") or _label_map_source(biovalidation_dir),
        "cell_type_label_map_schema": evidence.get("cell_type_label_map_schema"),
        "cell_type_label_cluster_key": evidence.get("cell_type_label_cluster_key"),
        "cluster_label_count": len(records),
    }
    return {key: value for key, value in fields.items() if value not in (None, "", [])}


def _save_case_sidecar(payload: dict[str, Any], biovalidation_dir: Path) -> Path:
    """Persist lightweight tabular data used by paired redraws."""
    return save_payload_sidecar(payload, biovalidation_dir)


def _top_gene_records_from_pdf(case_id: str, biovalidation_dir: Path) -> list[dict[str, Any]]:
    records = []
    for dim, gene, rho in _extract_top_gene_rows(case_id, biovalidation_dir):
        records.append({"dim": int(dim), "rank": 0, "gene": gene, "rho": float(rho)})
    return records


def _top_k_records_from_pdf(case_id: str, biovalidation_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in _extract_f_gene_labels(case_id, biovalidation_dir):
        for rank, label in enumerate(row):
            match = re.match(
                r"[dz](?P<dim>\d+)·(?P<gene>[^\s]+)\s+[ρr](?P<rho>[+-]\d+\.\d+)",
                label,
            )
            if not match:
                continue
            records.append({
                "dim": int(match.group("dim")),
                "rank": int(rank),
                "gene": match.group("gene"),
                "rho": float(match.group("rho")),
            })
    return records


def _case_sidecar_context(case_id: str) -> dict[str, Any]:
    from scccvgben.biovalidation import CASES

    case = CASES.get(case_id)
    if case is None:
        return {}
    return {
        "theme": case.theme,
        "accession": case.accession,
        "encoder": case.encoder,
        "condition_obs": case.condition_obs,
        "cell_type_obs": case.cell_type_obs,
        "condition_is_inferred": case.condition_obs is None,
        "cell_type_is_inferred": case.cell_type_obs is None,
    }


def _upgrade_cached_sidecar(
    case_id: str,
    biovalidation_dir: Path,
    sidecar: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Backfill old sidecars with the current tabular contract when possible."""
    upgraded = dict(sidecar)
    changed = False

    if int(upgraded.get("version", 0) or 0) < SIDECAR_VERSION:
        upgraded["version"] = SIDECAR_VERSION
        changed = True

    for key, value in _case_sidecar_context(case_id).items():
        if upgraded.get(key) != value:
            upgraded[key] = value
            changed = True

    if not upgraded.get("top_gene_rows"):
        rows = _top_gene_records_from_pdf(case_id, biovalidation_dir)
        if rows:
            upgraded["top_gene_rows"] = rows
            changed = True

    if not upgraded.get("top_k_genes_df"):
        records = _top_k_records_from_pdf(case_id, biovalidation_dir)
        if records:
            upgraded["top_k_genes_df"] = records
            changed = True

    label_evidence = _case_label_evidence(case_id, biovalidation_dir)
    for key, value in label_evidence.items():
        if value and upgraded.get(key) != value:
            upgraded[key] = value
            changed = True

    return upgraded, changed


def _load_case_sidecar(case_id: str, biovalidation_dir: Path) -> dict[str, Any] | None:
    path = _case_sidecar_path(case_id, biovalidation_dir)
    if not path.exists():
        return None
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    sidecar, changed = _upgrade_cached_sidecar(case_id, biovalidation_dir, sidecar)
    if changed:
        path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return sidecar


def _rebuild_case_sidecar_and_artifact(
    case_id: str,
    biovalidation_dir: Path,
    *,
    epochs: int,
    subsample_cells: int,
    top_k_genes: int,
) -> Path:
    """Recompute one case payload and export both single-case artifacts and sidecar."""
    from scccvgben.biovalidation import CASES
    from scccvgben.biovalidation.compute import run_case
    from scccvgben.biovalidation.compose import compose_case_figure

    if case_id not in CASES:
        raise SystemExit(f"unknown case id {case_id!r}")
    payload = run_case(
        CASES[case_id],
        epochs=epochs,
        subsample_cells=subsample_cells,
        top_k_genes=top_k_genes,
        enrichment_top_n_genes=100,
        enrichment_top_terms=8,
        silent=True,
    )
    compose_case_figure(payload, biovalidation_dir)
    return _save_case_sidecar(payload, biovalidation_dir)


def _paste_top_gene_panel(
    card: Image.Image,
    *,
    xy: tuple[int, int],
    size: tuple[int, int],
    case_id: str,
    biovalidation_dir: Path,
    sidecar: dict[str, Any] | None,
    label: str,
    title: str,
) -> None:
    """Draw top-gene cards instead of a shrunk raster table."""
    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(card)
    _draw_label(draw, (x + 6, y + 2), label, fill=_case_accent(case_id))
    _draw_panel_title(draw, (x + 70, y + 14), title, max_width=w - 86)

    rows = _extract_top_gene_rows(case_id, biovalidation_dir, sidecar)
    if not rows:
        draw.text((x + 34, y + 96), "top-gene table unavailable in cached PDF",
                  fill="#64748B", font=FONT_SM)
        return

    grid_x = x + 22
    grid_y = y + 62
    grid_w = w - 44
    grid_h = h - 82
    cols = 5
    n_rows = 2 if len(rows) > cols else 1
    gap = 10
    cell_w = (grid_w - (cols - 1) * gap) // cols
    cell_h = (grid_h - (n_rows - 1) * gap) // n_rows

    for i, (dim, gene, rho) in enumerate(rows[:10]):
        r = i // cols
        c = i % cols
        cx = grid_x + c * (cell_w + gap)
        cy = grid_y + r * (cell_h + gap)
        pos = rho >= 0
        fill = "#EFF6FF" if pos else "#FEF2F2"
        outline = "#93C5FD" if pos else "#FCA5A5"
        accent = "#0284C7" if pos else "#DC2626"
        draw.rounded_rectangle([cx, cy, cx + cell_w, cy + cell_h],
                               radius=12, fill=fill, outline=outline, width=2)
        draw.text((cx + 12, cy + 9), _latent_label(dim), fill="#334155", font=FONT_TABLE_DIM)
        gene_text = gene
        while len(gene_text) > 3 and _text_size(draw, gene_text, FONT_TABLE_GENE)[0] > cell_w - 24:
            gene_text = gene_text[:-1]
        if gene_text != gene:
            gene_text = gene_text.rstrip("_.:-") + "…"
        gw, _ = _text_size(draw, gene_text, FONT_TABLE_GENE)
        draw.text((cx + (cell_w - gw) // 2, cy + cell_h // 2 - 20),
                  gene_text, fill="#0F172A", font=FONT_TABLE_GENE)
        rho_text = f"r {rho:+.3f}"
        rw, _ = _text_size(draw, rho_text, FONT_TABLE_RHO)
        draw.text((cx + (cell_w - rw) // 2, cy + cell_h - 32),
                  rho_text, fill=accent, font=FONT_TABLE_RHO)


def _draw_fit_left(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    width: int,
    text: str,
    *,
    fill: str = "#0F172A",
) -> None:
    """Draw one left-aligned label, shortening only if it exceeds the gutter."""
    x, y = xy
    rendered = text
    while len(rendered) > 6 and _text_size(draw, rendered, FONT_GO)[0] > width:
        rendered = rendered[:-2]
    if rendered != text:
        rendered = rendered.rstrip(" ,;-") + "…"
    draw.text((x, y), rendered, fill=fill, font=FONT_GO)


def _lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * max(0.0, min(1.0, t))))


def _diverging_color(value: float, *, vmax: float = 2.0) -> tuple[int, int, int]:
    """Blue-white-red colour for robust median-z values."""
    v = max(-vmax, min(vmax, float(value)))
    blue = (37, 99, 235)
    white = (248, 250, 252)
    red = (220, 38, 38)
    if v < 0:
        t = (v + vmax) / vmax
        return (
            _lerp(blue[0], white[0], t),
            _lerp(blue[1], white[1], t),
            _lerp(blue[2], white[2], t),
        )
    t = v / vmax
    return (
        _lerp(white[0], red[0], t),
        _lerp(white[1], red[1], t),
        _lerp(white[2], red[2], t),
    )


def _sequential_color(case_id: str, value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    if vmax <= vmin:
        t = 0.55
    else:
        t = max(0.0, min(1.0, (float(value) - vmin) / (vmax - vmin)))
    rgba = matplotlib.colormaps[_case_cmap(case_id)](0.25 + 0.70 * t)
    return (
        int(round(255 * rgba[0])),
        int(round(255 * rgba[1])),
        int(round(255 * rgba[2])),
    )


def _rdbu_corr_color(value: float) -> tuple[int, int, int]:
    """RdBu_r colour for absolute latent-correlation heatmap cells."""
    t = max(0.0, min(1.0, float(value)))
    rgba = matplotlib.colormaps["RdBu_r"](t)
    return (
        int(round(255 * rgba[0])),
        int(round(255 * rgba[1])),
        int(round(255 * rgba[2])),
    )


def _contrast_text(rgb: tuple[int, int, int]) -> str:
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "white" if lum < 120 else "#0F172A"


def _draw_expression_colorbar(
    draw: ImageDraw.ImageDraw,
    *,
    anchor: tuple[int, int],
    case_id: str,
    size: tuple[int, int] = (16, 150),
    label: str = "expression",
) -> None:
    """Draw a small vertical expression colorbar with low/high tick labels.

    ``anchor`` is the bottom-right corner of the colorbar in panel-image
    coordinates. The bar extends upward and the caption sits above the bar.
    Vertical orientation saves horizontal space at the bottom-right corner.
    Font sizes match the surrounding panel labels (``FONT_GO_DOTPLOT``).
    """
    font = FONT_GO_DOTPLOT
    bar_w, bar_h = size
    x1, y1 = anchor
    x0, y0 = x1 - bar_w, y1 - bar_h
    cmap = matplotlib.colormaps[_case_cmap(case_id)]
    for j in range(bar_h):
        # top of the bar is the high end of the colormap.
        t = 1.0 - j / max(1, bar_h - 1)
        rgba = cmap(t)
        rgb = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        draw.line([(x0, y0 + j), (x1, y0 + j)], fill=rgb)
    draw.rectangle([x0, y0, x1, y1], outline="#94A3B8", width=1)
    high_w, high_h = _text_size(draw, "high", font)
    low_w, low_h = _text_size(draw, "low", font)
    draw.text((x0 - high_w - 6, y0 - 2), "high", fill="#0F172A", font=font)
    draw.text((x0 - low_w - 6, y1 - low_h + 1), "low", fill="#0F172A", font=font)
    if label:
        # Caption stacked above the bar to keep the footprint narrow.
        lab_w, lab_h = _text_size(draw, label, font)
        draw.text((x1 - lab_w, y0 - lab_h - 6), label,
                  fill="#475569", font=font)


def _draw_dotplot_key(
    draw: ImageDraw.ImageDraw,
    *,
    anchor: tuple[int, int],
    case_id: str,
    vmin: float,
    vmax: float,
    max_pct: float,
) -> None:
    """Color-gradient + 3-dot size key for the GO dot plot panel.

    The legend is stacked vertically: the color bar sits at the top and the
    overlap-size dots sit directly below it in the same column, so both keys
    align along a single vertical axis at the right of the panel.
    """
    font = FONT_GO_DOTPLOT
    x_anchor, y_anchor = anchor
    bar_w, bar_h = 14, 110
    # Compute the size-key block height first, then place the colorbar above
    # it so the whole legend ends at ``y_anchor`` (panel bottom).
    sizes = [1.0, 0.5, 0.25]
    radii = [4.5 + 13.0 * math.sqrt(s) for s in sizes]
    max_r = max(radii)
    gap_y = 6
    cap_text_size = "overlap"
    _, cap_size_h = _text_size(draw, cap_text_size, font)
    size_block_h = (
        6  # space between colorbar and size caption
        + cap_size_h
        + 6  # space between caption and first dot
        + 2 * max_r
        + 2 * (2 * max_r + gap_y)  # remaining two dots
    ) if max_pct > 0 else 0

    bar_x1 = x_anchor
    bar_y1 = y_anchor - size_block_h
    bar_x0 = bar_x1 - bar_w
    bar_y0 = bar_y1 - bar_h
    cmap = matplotlib.colormaps[_case_cmap(case_id)]
    for j in range(bar_h):
        t = 1.0 - j / max(1, bar_h - 1)
        rgba = cmap(0.25 + 0.70 * t)
        rgb = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        draw.line([(bar_x0, bar_y0 + j), (bar_x1, bar_y0 + j)], fill=rgb)
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], outline="#94A3B8", width=1)
    vmin_label = f"{float(vmin):.1f}"
    vmax_label = f"{float(vmax):.1f}"
    vmin_w, vmin_h = _text_size(draw, vmin_label, font)
    vmax_w, vmax_h = _text_size(draw, vmax_label, font)
    draw.text((bar_x0 - vmax_w - 6, bar_y0 - 2),
              vmax_label, fill="#0F172A", font=font)
    draw.text((bar_x0 - vmin_w - 6, bar_y1 - vmin_h + 1),
              vmin_label, fill="#0F172A", font=font)
    cap_text = "-log10(padj)"
    cap_w, cap_h = _text_size(draw, cap_text, font)
    draw.text((bar_x1 - cap_w, bar_y0 - cap_h - 6),
              cap_text, fill="#475569", font=font)

    # Size key directly under the colorbar, sharing the same vertical axis.
    if max_pct > 0:
        col_x = (bar_x0 + bar_x1) / 2
        sw, _ = _text_size(draw, cap_text_size, font)
        cap_y = bar_y1 + 6
        draw.text((col_x - sw / 2, cap_y), cap_text_size, fill="#475569", font=font)
        cy = cap_y + cap_size_h + 6 + max_r
        for r in radii:
            draw.ellipse([col_x - r, cy - r, col_x + r, cy + r],
                         fill="#CBD5E1", outline="#475569", width=1)
            cy += 2 * max_r + gap_y


def _paste_condition_summary_panel(
    card: Image.Image,
    *,
    xy: tuple[int, int],
    size: tuple[int, int],
    case_id: str,
    sidecar: dict[str, Any] | None,
    label: str,
    title: str,
) -> bool:
    """Draw condition × latent summaries directly from sidecar rows."""
    data = (sidecar or {}).get("condition_latent", {})
    dims = data.get("dims") or []
    conditions = data.get("conditions") or []
    records = data.get("records") or []
    if not dims or not conditions or not records:
        return False

    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(card)
    _draw_label(draw, (x + 6, y + 2), label, fill=_case_accent(case_id))
    _draw_panel_title(draw, (x + 70, y + 14), title, max_width=w - 86)

    rec_map = {(int(r["dim"]), str(r["condition"])): r for r in records}
    labels = [str(c["label"]) for c in conditions]
    counts = [int(c.get("n", 0)) for c in conditions]

    left = x + 48
    right = x + w - 18
    top = y + 76
    bottom = y + h - 58
    grid_w = right - left
    grid_h = bottom - top
    n_cols = max(1, len(labels))
    n_rows = max(1, len(dims))
    gap = 4
    cell_w = max(18, (grid_w - gap * (n_cols - 1)) // n_cols)
    cell_h = max(26, (grid_h - gap * (n_rows - 1)) // n_rows)

    # Count bars make category imbalance visible without adding a legend.
    max_n = max(counts) if counts else 1
    for j, (label, n) in enumerate(zip(labels, counts, strict=False)):
        cx = left + j * (cell_w + gap)
        bh = max(3, round(24 * n / max_n))
        draw.rounded_rectangle([cx + 5, top - 34 + (24 - bh), cx + cell_w - 5, top - 10],
                               radius=3, fill="#CBD5E1")
        txt = label
        while len(txt) > 2 and _text_size(draw, txt, FONT_GO)[0] > cell_w:
            txt = txt[:-1]
        if txt != label:
            txt = txt.rstrip("_-.") + "…"
        tw, _ = _text_size(draw, txt, FONT_GO)
        draw.text((cx + max(0, (cell_w - tw) // 2), bottom + 8), txt,
                  fill="#334155", font=FONT_GO)

    for i, dim in enumerate(dims):
        ry = top + i * (cell_h + gap)
        draw.text((x + 17, ry + cell_h // 2 - 7), _latent_label(dim), fill="#334155", font=FONT_TABLE_DIM)
        for j, label in enumerate(labels):
            cx = left + j * (cell_w + gap)
            r = rec_map.get((int(dim), label))
            if not r:
                fill = (241, 245, 249)
                text = "–"
            else:
                value = float(r.get("median_z", 0.0))
                fill = _diverging_color(value)
                text = f"{value:+.1f}"
            draw.rounded_rectangle([cx, ry, cx + cell_w, ry + cell_h],
                                   radius=7, fill=fill, outline="#E2E8F0", width=1)
            tw, th = _text_size(draw, text, FONT_GO)
            draw.text((cx + (cell_w - tw) // 2, ry + (cell_h - th) // 2 - 1),
                      text, fill=_contrast_text(fill), font=FONT_GO)

    draw.text((x + 20, y + h - 28), "cell-count bars · median z-score",
              fill="#64748B", font=FONT_GO)
    return True


def _paste_latent_corr_panel(
    card: Image.Image,
    *,
    xy: tuple[int, int],
    size: tuple[int, int],
    case_id: str,
    sidecar: dict[str, Any] | None,
    label: str,
    title: str,
) -> bool:
    """Draw D/L from cached sidecar matrix using a RdBu heatmap palette."""
    matrix = (sidecar or {}).get("latent_corr") or []
    if not matrix:
        return False
    try:
        corr = [[float(v) for v in row] for row in matrix]
    except (TypeError, ValueError):
        return False
    n = len(corr)
    if n == 0 or any(len(row) != n for row in corr):
        return False

    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(card)
    _draw_label(draw, (x + 6, y + 2), label, fill=_case_accent(case_id))
    _draw_panel_title(draw, (x + 70, y + 14), title, max_width=w - 86)

    top = y + 62
    bottom_gutter = 34
    left_gutter = 42
    right_gutter = 36
    available_w = w - left_gutter - right_gutter
    available_h = h - (top - y) - bottom_gutter
    cell = max(8, min(available_w, available_h) // n)
    grid = cell * n
    gx = x + left_gutter + max(0, (available_w - grid) // 2)
    gy = top + max(0, (available_h - grid) // 2)

    for i, row in enumerate(corr):
        for j, value in enumerate(row):
            fill = _rdbu_corr_color(value)
            x0 = gx + j * cell
            y0 = gy + i * cell
            draw.rectangle(
                [x0, y0, x0 + cell - 1, y0 + cell - 1],
                fill=fill,
                outline="#F8FAFC",
            )

    tick_every = 1 if n <= 10 else 2
    for idx in range(0, n, tick_every):
        tick = _latent_label(idx)
        tx = gx + idx * cell + cell / 2
        tw, th = _text_size(draw, tick, FONT_GO)
        draw.text((round(tx - tw / 2), gy + grid + 6),
                  tick, fill="#334155", font=FONT_GO)
        draw.text((gx - tw - 7, round(gy + idx * cell + (cell - th) / 2)),
                  tick, fill="#334155", font=FONT_GO)

    bar_x0 = gx + grid + 12
    bar_h = min(grid, 118)
    bar_y0 = gy + max(0, (grid - bar_h) // 2)
    for jj in range(bar_h):
        value = 1.0 - jj / max(1, bar_h - 1)
        draw.line(
            [(bar_x0, bar_y0 + jj), (bar_x0 + 10, bar_y0 + jj)],
            fill=_rdbu_corr_color(value),
        )
    draw.rectangle([bar_x0, bar_y0, bar_x0 + 10, bar_y0 + bar_h],
                   outline="#94A3B8", width=1)
    draw.text((bar_x0 + 15, bar_y0 - 2), "1", fill="#334155", font=FONT_GO)
    draw.text((bar_x0 + 15, bar_y0 + bar_h - 13), "0", fill="#334155", font=FONT_GO)
    draw.text((gx, y + h - 18), "|corr|", fill="#64748B", font=FONT_GO)
    return True


def _pretty_go_term(term: str) -> str:
    raw = str(term)
    for prefix in ("GOBP_", "GO_BP_", "REACTOME_", "KEGG_"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    return raw.replace("_", " ").lower()


def _paste_go_summary_panel(
    card: Image.Image,
    *,
    xy: tuple[int, int],
    size: tuple[int, int],
    case_id: str,
    sidecar: dict[str, Any] | None,
    label: str,
    title: str,
    max_terms: int = 14,
) -> bool:
    """Draw panel H directly from enrichment rows without raster crop truncation."""
    records = (sidecar or {}).get("enrichment") or []
    if not records:
        return False

    # Keep the strongest row for each term × dim and select best terms.
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for r in records:
        try:
            term = str(r["term"])
            dim = int(r["dim"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (term, dim)
        current = best.get(key)
        if current is None or float(r.get("padj", 1.0)) < float(current.get("padj", 1.0)):
            best[key] = r
    if not best:
        return False

    term_scores: dict[str, float] = {}
    for term, _dim in best:
        score = float(best[(term, _dim)].get("neg_log10_padj", 0.0) or 0.0)
        term_scores[term] = max(term_scores.get(term, 0.0), score)
    terms = sorted(term_scores, key=lambda t: (-term_scores[t], _pretty_go_term(t)))[:max_terms]
    dims = list(range(10))

    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(card)
    _draw_label(draw, (x + 6, y + 2), label, fill=_case_accent(case_id))
    _draw_panel_title(draw, (x + 70, y + 14), title, max_width=w - 86)

    plot_top = y + 70
    plot_bottom = y + h - 46
    label_w = 580
    plot_left = x + 24 + label_w
    # Reserve a right gutter so the vertical color/size key sits outside the
    # data area and never overlaps with the rightmost dim column. The key is
    # now stacked vertically (colorbar above size dots) so a narrower gutter
    # suffices.
    plot_right = x + w - 118
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top
    row_h = plot_h / max(1, len(terms))
    col_w = plot_w / max(1, len(dims))

    vals = [float(r.get("neg_log10_padj", 0.0) or 0.0) for r in best.values()]
    pcts = [float(r.get("percent", 0.0) or 0.0) for r in best.values()]
    vmin, vmax = min(vals), max(vals)
    max_pct = max(pcts) if pcts else 0.01
    max_pct = max(max_pct, 0.01)

    # Grid.
    for j, dim in enumerate(dims):
        gx = plot_left + j * col_w + col_w / 2
        draw.line([(gx, plot_top), (gx, plot_bottom)], fill="#EEF2F7", width=1)
        tick = _latent_label(dim)
        tw, _ = _text_size(draw, tick, FONT_GO_DOTPLOT)
        draw.text((round(gx - tw / 2), y + h - 34), tick,
                  fill="#0F172A", font=FONT_GO_DOTPLOT)
    for i, term in enumerate(terms):
        gy = plot_top + i * row_h + row_h / 2
        draw.line([(plot_left, gy), (plot_right, gy)], fill="#EEF2F7", width=1)
        label = _pretty_go_term(term)
        while len(label) > 8 and _text_size(draw, label, FONT_GO_DOTPLOT)[0] > label_w - 10:
            label = label[:-2]
        if label != _pretty_go_term(term):
            label = label.rstrip(" ,;-") + "…"
        _, label_h = _text_size(draw, label, FONT_GO_DOTPLOT)
        draw.text((x + 24, round(gy - label_h / 2)), label,
                  fill="#111827", font=FONT_GO_DOTPLOT)

    # Points — radius bumped so dots in the 10-column dotplot remain visible
    # even when many terms compress row height; matches scale of the larger
    # term/tick labels above.
    for i, term in enumerate(terms):
        for j, dim in enumerate(dims):
            r = best.get((term, dim))
            if not r:
                continue
            pct = float(r.get("percent", 0.0) or 0.0)
            score = float(r.get("neg_log10_padj", 0.0) or 0.0)
            radius = 3.6 + 11.0 * math.sqrt(max(0.0, pct) / max_pct)
            cx = plot_left + j * col_w + col_w / 2
            cy = plot_top + i * row_h + row_h / 2
            fill = _sequential_color(case_id, score, vmin, vmax)
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                         fill=fill, outline="white", width=1)

    draw.line([(plot_left, plot_bottom), (plot_right, plot_bottom)],
              fill="#CBD5E1", width=1)
    # Color + size key replaces the older free-text legend so readers can
    # quantitatively read the dot plot without referring to the caption.
    _draw_dotplot_key(
        draw,
        anchor=(x + w - 18, plot_bottom),
        case_id=case_id,
        vmin=vmin,
        vmax=vmax,
        max_pct=max_pct,
    )
    return True


def _gene_grid_image(
    src: Image.Image,
    crop: tuple[int, int, int, int],
    labels: list[list[str]],
    size: tuple[int, int],
    case_id: str,
) -> Image.Image:
    """Rebuild panel F as a clean 3×5 grid.

    Cached single-case figures contain the right biological mini-UMAPs, but
    their matplotlib titles collide horizontally.  This helper crops only the
    mini-UMAP image bands and redraws the labels in local Arial, preserving the
    cached results while making the panel readable.
    """
    raw = src.crop(crop)
    raw_draw = ImageDraw.Draw(raw)
    # Remove the original matplotlib titles, which are the source of the
    # collisions.  Keep the scatter panels untouched and redraw clean labels
    # after the whole cached panel is scaled.
    # The cached single-case PNG can be emitted at different pixel sizes
    # (2304×2304 in the current pipeline, historically 3200×3200).  The title
    # shelves below are therefore expressed in the 2090×1010 logical crop
    # space and scaled to the actual crop height before masking.  Using fixed
    # pixel bands here was what let old row-2/row-3 matplotlib titles survive
    # and collide with the redesigned Arial labels in the paired figures.
    scale_y = raw.height / 1010
    for y0, y1 in [(0, 82), (318, 414), (650, 762)]:
        y0s = max(0, round(y0 * scale_y))
        y1s = min(raw.height, round(y1 * scale_y))
        if y1s > y0s:
            raw_draw.rectangle([0, y0s, raw.width, y1s], fill="white")

    # Trim the cached F crop's left/right whitespace before fitting.  The
    # matplotlib source carries ~6–9% horizontal padding around the 5×3
    # mini-UMAP block; `thumbnail` preserved that padding inside the panel and
    # shrank the actual scatter content.  Detect the non-white horizontal bbox
    # and re-crop only the X axis — vertical bands are preserved so the
    # row-shelf label positions below (`[6, 338, 680]` in 1010-tall logical
    # space) remain valid for the redrawn Arial gene labels.
    gray = raw.convert("L")
    non_white = gray.point(lambda v: 0 if v >= 248 else 255)
    bbox = non_white.getbbox()
    if bbox is not None:
        bx0, _, bx1, _ = bbox
        margin = max(2, round(raw.width * 0.004))
        bx0 = max(0, bx0 - margin)
        bx1 = min(raw.width, bx1 + margin)
        if bx1 > bx0:
            raw = raw.crop((bx0, 0, bx1, raw.height))

    out_w, out_h = size
    out = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(out)

    cols, rows = 5, 3
    # Reserve a right gutter for the vertical expression colorbar so the mini-
    # UMAP block never sits underneath the legend.
    right_gutter = 70
    raw.thumbnail((max(64, out_w - right_gutter), out_h), Image.Resampling.LANCZOS)
    # If the height-bound thumbnail leaves horizontal slack, stretch the
    # block (≤ 12 %) so the inner column gaps widen and the white area
    # between the rightmost column and the colorbar disappears.
    avail_w = out_w - right_gutter - 6
    if raw.width < avail_w:
        stretch_w = min(avail_w, round(raw.width * 1.12))
        if stretch_w > raw.width:
            raw = raw.resize((stretch_w, raw.height), Image.Resampling.LANCZOS)
    # Left-align so the gutter is fully on the right side; no centering.
    origin_x = 4
    origin_y = (out_h - raw.height) // 2
    out.paste(raw, (origin_x, origin_y))

    for r in range(rows):
        for c in range(cols):
            cell_w = raw.width // cols
            tx = origin_x + c * cell_w
            # Source-row title bands start near y=0, 332, and 674 in the
            # 2090×1010 cached F crop; after thumbnailing they remain clean
            # label shelves for the overlaid Arial labels.
            shelf_y = origin_y + round([6, 338, 680][r] * (raw.height / 1010))
            text = labels[r][c] if r < len(labels) and c < len(labels[r]) else ""
            _draw_fit_centered(draw, (tx + 3, shelf_y), cell_w - 6, text)

    _draw_expression_colorbar(
        draw,
        anchor=(out_w - 8, out_h - 8),
        case_id=case_id,
        size=(16, 150),
        label="expression",
    )

    return out


def _paste_gene_grid_panel(
    card: Image.Image,
    src: Image.Image,
    *,
    xy: tuple[int, int],
    size: tuple[int, int],
    crop: tuple[int, int, int, int],
    case_id: str,
    biovalidation_dir: Path,
    sidecar: dict[str, Any] | None,
    label: str,
    title: str,
) -> None:
    """Paste the latent-gene panel with collision-free labels."""
    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(card)
    _draw_label(draw, (x + 6, y + 2), label, fill=_case_accent(case_id))
    _draw_panel_title(draw, (x + 70, y + 14), title, max_width=w - 86)
    image_box = (x + 18, y + 52, w - 36, h - 66)
    labels = _extract_f_gene_labels(case_id, biovalidation_dir, sidecar)
    if labels:
        fitted = _gene_grid_image(
            src, crop, labels, (image_box[2], image_box[3]), case_id
        )
    else:
        fitted = _fit_crop(src, crop, (image_box[2], image_box[3]))
    card.paste(fitted, (image_box[0], image_box[1]))


def _paste_go_panel(
    card: Image.Image,
    src: Image.Image,
    *,
    xy: tuple[int, int],
    size: tuple[int, int],
    plot_crop: tuple[int, int, int, int],
    case_id: str,
    label: str,
    title: str,
) -> None:
    """Paste H with readable term labels while preserving dotplot aspect ratio."""
    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(card)
    _draw_label(draw, (x + 6, y + 2), label, fill=_case_accent(case_id))
    _draw_panel_title(draw, (x + 70, y + 14), title, max_width=w - 86)

    image_x = x + 18
    image_y = y + 52
    image_w = w - 36
    image_h = h - 68
    terms = _H_TERM_LABELS.get(case_id, [])
    if terms:
        max_term_w = max(_text_size(draw, term, FONT_GO)[0] for term in terms)
        label_w = max(260, min(500, max_term_w + 20))
    else:
        label_w = 300
    gutter = 14
    plot_w = image_w - label_w - gutter

    plot = src.crop(plot_crop)
    # Keep the cached dotplot at its natural aspect ratio.  Stretching the
    # source raster makes points and axis text misleading; the separate Arial
    # gutter below carries readable GO labels without distorting the data.
    plot.thumbnail((plot_w, image_h - 4), Image.Resampling.LANCZOS)
    plot_x = image_x + label_w + gutter + (plot_w - plot.width) // 2
    plot_y = image_y + 4

    if terms:
        step = plot.height / max(len(terms), 1)
        for i, term in enumerate(terms):
            ty = plot_y + round(i * step + max(0, step - 12) / 2)
            _draw_fit_left(draw, (image_x, ty), label_w - 8, term, fill="#111827")
    card.paste(plot, (plot_x, plot_y))


def _case_subtitle(case_id: str) -> str:
    from scccvgben.biovalidation import CASES

    case = CASES[case_id]
    accession = case.accession
    if case_id in {"SD", "UCB", "IR"}:
        accession = f"Reference subset {case_id}"
    return f"{accession} · theme={case.theme} · encoder={case.encoder}"


def _role_labels(side: str) -> dict[str, str]:
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    offset = 0 if side == "left" else len(PANEL_ROLE_ORDER)
    return {
        role: chr(ord("A") + offset + i)
        for i, role in enumerate(PANEL_ROLE_ORDER)
    }


def _main_role_labels(side: str) -> dict[str, str]:
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    offset = 0 if side == "left" else len(MAIN_PANEL_ROLE_ORDER)
    return {
        role: chr(ord("A") + offset + i)
        for i, role in enumerate(MAIN_PANEL_ROLE_ORDER)
    }


def _role_titles(case: Any) -> dict[str, str]:
    condition_title = "Condition / batch UMAP" if case.condition_obs else "Inferred group UMAP"
    summary_title = "Condition × latent-z summary" if case.condition_obs else "Group × latent-z summary"
    cell_title = "Cell-type UMAP" if case.cell_type_obs else "Marker-labelled cell-state UMAP"
    return {
        "condition_umap": condition_title,
        "cell_type_umap": cell_title,
        "latent_d0_umap": "Latent z0 UMAP",
        "latent_corr": "Latent self-corr",
        "latent_gene_grid": "Latent gene mini-UMAPs",
        "condition_latent_summary": summary_title,
        "top_gene_table": "Top gene per latent z",
        "go_enrichment": "GO biological-process enrichment",
    }


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _label_evidence_summary_for_cases(
    case_ids: list[str],
    biovalidation_dir: Path,
) -> dict[str, Any]:
    """Summarise cell-type label provenance without duplicating full records."""
    source = _label_map_source(biovalidation_dir)
    cases: dict[str, Any] = {}
    for case_id in case_ids:
        evidence = _case_label_evidence(case_id, biovalidation_dir)
        records = evidence.get("cluster_label_evidence") or []
        if evidence or records:
            cases[case_id] = {
                key: value
                for key, value in {
                    "cell_type_label_source": evidence.get("cell_type_label_source"),
                    "cluster_key": evidence.get("cell_type_label_cluster_key"),
                    "cluster_label_count": len(records),
                    "sidecar_record": (
                        f"{_rel(_case_sidecar_path(case_id, biovalidation_dir))}"
                        "#cluster_label_evidence"
                        if records else None
                    ),
                }.items()
                if value not in (None, "", [])
            }
    if not source and not cases:
        return {}
    return {
        key: value
        for key, value in {
            "label_map": source,
            "label_map_schema": (
                _case_label_evidence(case_ids[0], biovalidation_dir).get("cell_type_label_map_schema")
                if case_ids else None
            ),
            "cases": cases,
        }.items()
        if value not in (None, "", {})
    }


def _source_manifest_payload(
    *,
    out_stem: str,
    pair_label: str,
    left_case: str,
    right_case: str,
    panels: list[dict[str, Any]],
    biovalidation_dir: Path,
    png_path: Path,
    pdf_path: Path,
    label_manifest_path: Path,
    supplemental_outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Describe the source-first contract for one paired bio-validation figure."""
    source_manifest_path = biovalidation_dir / f"{out_stem}.source_manifest.json"
    case_ids = [left_case, right_case]
    metadata_paths = [
        _rel(label_manifest_path),
        _rel(source_manifest_path),
        *[
            _rel(_case_sidecar_path(case_id, biovalidation_dir))
            for case_id in case_ids
        ],
    ]
    label_map = _label_map_source(biovalidation_dir)
    if label_map is not None:
        metadata_paths.append(label_map)
    if supplemental_outputs and supplemental_outputs.get("manifest"):
        metadata_paths.append(supplemental_outputs["manifest"])

    manifest = {
        "version": SOURCE_MANIFEST_VERSION,
        "figure_id": out_stem,
        "pair_label": pair_label,
        "source_contract": (
            "scripts, sidecar/manifest metadata, and LaTeX are primary; "
            "generated PNG/PDF binaries are local QA evidence"
        ),
        "primary_deliverables": {
            "script": _rel(REPO_ROOT / "scripts" / "make_biovalidation_pairs.py"),
            "latex": _rel(MANUSCRIPT_SOURCE),
            "metadata": metadata_paths,
        },
        "qa_artifacts": {
            "generated_pair_binaries": {
                "role": "qa_evidence_only",
                "commit_policy": "do_not_commit",
                "paths": [_rel(png_path), _rel(pdf_path)],
            },
            "cached_case_render_inputs": [
                {
                    "case_id": case_id,
                    "png": _rel(biovalidation_dir / f"fig_biovalidation_case_{case_id}.png"),
                    "pdf": _rel(biovalidation_dir / f"fig_biovalidation_case_{case_id}.pdf"),
                    "role": "local_render_cache_for_panel_crops",
                    "commit_policy": "do_not_commit",
                }
                for case_id in case_ids
            ],
        },
        "detail_sources": [
            {
                "case_id": case_id,
                "sidecar": _rel(_case_sidecar_path(case_id, biovalidation_dir)),
                "records": [
                    "condition_latent",
                    "cluster_label_evidence",
                    "top_gene_rows",
                    "top_k_genes_df",
                    "latent_corr",
                    "enrichment",
                ],
                "role": "traceable_detail_payload_for_main_or_supplement_panels",
            }
            for case_id in case_ids
        ],
        "recompute_policy": {
            "default": "reuse_cached_artifacts_and_sidecars",
            "training_or_result_recompute": "disabled_by_default",
            "explicit_flags_required": ["--allow-recompute", "--rebuild-sidecars"],
        },
        "cases": {
            "left": left_case,
            "right": right_case,
        },
        "panel_count": len(panels),
        "label_sequence": [panel["global_label"] for panel in panels],
        "label_manifest": _rel(label_manifest_path),
        "manuscript_label": out_stem,
    }
    label_evidence = _label_evidence_summary_for_cases(case_ids, biovalidation_dir)
    if label_evidence:
        manifest["label_evidence"] = label_evidence
    if supplemental_outputs is not None:
        manifest["supplemental_outputs"] = supplemental_outputs
    return manifest


def _write_source_manifest(
    *,
    out_stem: str,
    pair_label: str,
    left_case: str,
    right_case: str,
    panels: list[dict[str, Any]],
    biovalidation_dir: Path,
    png_path: Path,
    pdf_path: Path,
    label_manifest_path: Path,
    supplemental_outputs: dict[str, str] | None = None,
) -> Path:
    """Persist source/provenance policy beside the panel-label manifest."""
    path = biovalidation_dir / f"{out_stem}.source_manifest.json"
    manifest = _source_manifest_payload(
        out_stem=out_stem,
        pair_label=pair_label,
        left_case=left_case,
        right_case=right_case,
        panels=panels,
        biovalidation_dir=biovalidation_dir,
        png_path=png_path,
        pdf_path=pdf_path,
        label_manifest_path=label_manifest_path,
        supplemental_outputs=supplemental_outputs,
    )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def _panel_manifest_record(
    *,
    figure_id: str,
    global_label: str,
    visual_slot: int,
    side: str,
    case_id: str,
    role: str,
    title: str,
    data_source: str,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "global_label": global_label,
        "visual_slot": visual_slot,
        "side": side,
        "case_id": case_id,
        "role": role,
        "source_semantic_id": SOURCE_ROLE_LETTERS[role],
        "source_role_letter": SOURCE_ROLE_LETTERS[role],
        "title": title,
        "data_source": data_source,
    }


def _build_main_case_card(
    case_id: str,
    biovalidation_dir: Path,
    *,
    side: str,
    figure_id: str,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Convert a cached single-case PNG into a concise manuscript card."""
    from scccvgben.biovalidation import CASES

    case = CASES[case_id]
    sidecar = _load_case_sidecar(case_id, biovalidation_dir)
    src_path = biovalidation_dir / f"fig_biovalidation_case_{case_id}.png"
    sidecar_path = _case_sidecar_path(case_id, biovalidation_dir)
    src = Image.open(src_path).convert("RGB")
    sw, sh = src.size
    sx, sy = sw / 3200.0, sh / 3200.0

    def box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        return (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))

    card = Image.new("RGB", (CARD_W, MAIN_CARD_H), "white")

    labels = _main_role_labels(side)
    titles = _role_titles(case)
    accent = _case_accent(case_id)
    visual_offset = 0 if side == "left" else len(MAIN_PANEL_ROLE_ORDER)
    manifest: list[dict[str, Any]] = []

    def record(role: str, data_source: str) -> None:
        manifest.append(_panel_manifest_record(
            figure_id=figure_id,
            global_label=labels[role],
            visual_slot=visual_offset + MAIN_PANEL_ROLE_ORDER.index(role) + 1,
            side=side,
            case_id=case_id,
            role=role,
            title=titles[role],
            data_source=data_source,
        ))

    pad = 18
    inner_w = CARD_W - 2 * pad
    top_y = 16
    top_h = 415
    col_gap = 16
    col_w = (inner_w - 2 * col_gap) // 3

    top_crops = [
        ("condition_umap", box(410, 275, 1115, 765)),
        ("latent_d0_umap", box(1780, 275, 2505, 765)),
    ]
    for i, (role, crop) in enumerate(top_crops):
        _paste_panel(
            card, src,
            xy=(pad + i * (col_w + col_gap), top_y),
            size=(col_w, top_h),
            crop=crop,
            label=labels[role],
            title=titles[role],
            accent=accent,
        )
        record(role, f"{_rel(src_path)}#crop:{role}")

    summary_x = pad + 2 * (col_w + col_gap)
    if not _paste_condition_summary_panel(
        card,
        xy=(summary_x, top_y),
        size=(col_w, top_h),
        case_id=case_id,
        sidecar=sidecar,
        label=labels["condition_latent_summary"],
        title=titles["condition_latent_summary"],
    ):
        _paste_panel(
            card, src,
            xy=(summary_x, top_y),
            size=(col_w, top_h),
            crop=box(2460, 1020, 3195, 1980),
            label=labels["condition_latent_summary"],
            title=titles["condition_latent_summary"],
            accent=accent,
        )
    record("condition_latent_summary", f"{_rel(sidecar_path)}#condition_latent")

    go_y = top_y + top_h + 18
    go_h = MAIN_CARD_H - go_y - pad
    if not _paste_go_summary_panel(
        card,
        xy=(pad, go_y),
        size=(inner_w, go_h),
        case_id=case_id,
        sidecar=sidecar,
        label=labels["go_enrichment"],
        title="GO biological-process narrative",
        max_terms=8,
    ):
        _paste_go_panel(
            card, src,
            xy=(pad, go_y),
            size=(inner_w, go_h),
            plot_crop=box(650, 2580, 2950, 3120),
            case_id=case_id,
            label=labels["go_enrichment"],
            title="GO biological-process narrative",
        )
    record("go_enrichment", f"{_rel(sidecar_path)}#enrichment")
    return card, manifest


def _build_case_card(
    case_id: str,
    biovalidation_dir: Path,
    *,
    side: str,
    figure_id: str,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Convert a cached single-case PNG into a compact publication card."""
    from scccvgben.biovalidation import CASES

    case = CASES[case_id]
    sidecar = _load_case_sidecar(case_id, biovalidation_dir)
    src_path = biovalidation_dir / f"fig_biovalidation_case_{case_id}.png"
    sidecar_path = _case_sidecar_path(case_id, biovalidation_dir)
    src = Image.open(src_path).convert("RGB")
    sw, sh = src.size
    sx, sy = sw / 3200.0, sh / 3200.0

    def box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        return (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))

    card = Image.new("RGB", (CARD_W, CARD_H), "white")

    labels = _role_labels(side)
    titles = _role_titles(case)
    accent = _case_accent(case_id)
    visual_offset = 0 if side == "left" else len(PANEL_ROLE_ORDER)
    manifest: list[dict[str, Any]] = []

    def record(role: str, data_source: str) -> None:
        panel = _panel_manifest_record(
            figure_id=figure_id,
            global_label=labels[role],
            visual_slot=visual_offset + PANEL_ROLE_ORDER.index(role) + 1,
            side=side,
            case_id=case_id,
            role=role,
            title=titles[role],
            data_source=data_source,
        )
        if role == "cell_type_umap":
            panel.update(_case_label_manifest_fields(case_id, biovalidation_dir, sidecar))
            if panel.get("cluster_label_count", 0):
                panel["label_evidence_source"] = f"{_rel(sidecar_path)}#cluster_label_evidence"
        manifest.append(panel)

    pad = 18
    inner_w = CARD_W - 2 * pad
    top_y = 16
    top_h = 330
    col_gap = 14
    col_w = (inner_w - 3 * col_gap) // 4
    top_crops = [
        ("condition_umap", box(410, 275, 1115, 765)),
        ("cell_type_umap", box(1095, 275, 1770, 765)),
        ("latent_d0_umap", box(1780, 275, 2505, 765)),
        ("latent_corr", box(2515, 245, 3195, 750)),
    ]
    for i, (role, crop) in enumerate(top_crops):
        xy = (pad + i * (col_w + col_gap), top_y)
        size = (col_w, top_h)
        data_source = f"{_rel(src_path)}#crop:{role}"
        if role == "latent_corr" and _paste_latent_corr_panel(
            card,
            xy=xy,
            size=size,
            case_id=case_id,
            sidecar=sidecar,
            label=labels[role],
            title=titles[role],
        ):
            data_source = f"{_rel(sidecar_path)}#latent_corr"
        else:
            if role == "cell_type_umap" and _paste_cell_type_panel(
                card, src,
                xy=xy,
                size=size,
                crop=crop,
                case_id=case_id,
                biovalidation_dir=biovalidation_dir,
                sidecar=sidecar,
                label=labels[role],
                title=titles[role],
                accent=accent,
            ):
                data_source = (
                    f"{_rel(src_path)}#crop:{role}+"
                    f"{_rel(sidecar_path)}#cluster_label_evidence"
                )
            else:
                _paste_panel(
                    card, src,
                    xy=xy,
                    size=size,
                    crop=crop,
                    label=labels[role],
                    title=titles[role],
                    accent=accent,
                )
        record(role, data_source)

    mid_y = top_y + top_h + 18
    # Source crop aspect for panels E/M is ~2.07; give the gene grid a little
    # more horizontal room so labels and mini-UMAPs do not feel squeezed.
    f_h = 830
    gene_x = 10
    gene_w = CARD_W - 2 * gene_x
    _paste_gene_grid_panel(
        card, src,
        xy=(gene_x, mid_y),
        size=(gene_w, f_h),
        crop=box(405, 850, 2495, 1860),
        case_id=case_id,
        biovalidation_dir=biovalidation_dir,
        sidecar=sidecar,
        label=labels["latent_gene_grid"],
        title=titles["latent_gene_grid"],
    )
    record("latent_gene_grid", f"{_rel(src_path)}+{_rel(sidecar_path)}#top_k_genes_df")

    summary_y = mid_y + f_h + 18
    # Panel G is intrinsically tall and was previously squeezed into a
    # short/wide slot, leaving large side gutters around the violin plot.
    # Give this row more height and a narrower G card while keeping F and H
    # in the same vertical budget.
    summary_h = 420
    g_w = 560
    if not _paste_condition_summary_panel(
        card,
        xy=(pad, summary_y),
        size=(g_w, summary_h),
        case_id=case_id,
        sidecar=sidecar,
        label=labels["condition_latent_summary"],
        title=titles["condition_latent_summary"],
    ):
        _paste_panel(
            card, src,
            xy=(pad, summary_y),
            size=(g_w, summary_h),
            crop=box(2460, 1020, 3195, 1980),
            label=labels["condition_latent_summary"],
            title=titles["condition_latent_summary"],
            accent=accent,
        )
    record("condition_latent_summary", f"{_rel(sidecar_path)}#condition_latent")

    _paste_top_gene_panel(
        card,
        xy=(pad + g_w + col_gap, summary_y),
        size=(inner_w - g_w - col_gap, summary_h),
        case_id=case_id,
        biovalidation_dir=biovalidation_dir,
        sidecar=sidecar,
        label=labels["top_gene_table"],
        title=titles["top_gene_table"],
    )
    record("top_gene_table", f"{_rel(sidecar_path)}#top_gene_rows")

    h_y = summary_y + summary_h + 18
    if not _paste_go_summary_panel(
        card,
        xy=(pad, h_y),
        size=(inner_w, CARD_H - h_y - pad),
        case_id=case_id,
        sidecar=sidecar,
        label=labels["go_enrichment"],
        title=titles["go_enrichment"],
    ):
        _paste_go_panel(
            card, src,
            xy=(pad, h_y),
            size=(inner_w, CARD_H - h_y - pad),
            # Crop away the original title, x-axis prose label, and right-side
            # colour/size legends; those encodings belong in the manuscript
            # caption, while the paired card should spend pixels on the term ×
            # latent-dim evidence itself.
            plot_crop=box(650, 2580, 2950, 3120),
            case_id=case_id,
            label=labels["go_enrichment"],
            title=titles["go_enrichment"],
        )
    record("go_enrichment", f"{_rel(sidecar_path)}#enrichment")
    return card, manifest


def _write_label_manifest(
    *,
    out_stem: str,
    pair_label: str,
    left_case: str,
    right_case: str,
    panels: list[dict[str, Any]],
    biovalidation_dir: Path,
    png_path: Path,
    pdf_path: Path,
    role_order: list[str] | None = None,
    label_policy: str | None = None,
    figure_kind: str = "main",
    parent_figure_id: str | None = None,
    supplemental_outputs: dict[str, str] | None = None,
    moved_to_supplement: list[str] | None = None,
    source_manifest_path: Path | None = None,
) -> Path:
    """Persist a stable manifest mapping visible labels to biological roles."""
    role_order = role_order or PANEL_ROLE_ORDER
    labels = [panel["global_label"] for panel in panels]
    expected = [chr(ord("A") + i) for i in range(len(role_order) * 2)]
    if labels != expected:
        raise ValueError(f"{out_stem}: non-sequential labels {labels!r}; expected {expected!r}")
    if len(labels) != len(set(labels)):
        raise ValueError(f"{out_stem}: duplicate panel labels {labels!r}")

    outputs = {
        "png": _rel(png_path),
        "pdf": _rel(pdf_path),
    }
    if source_manifest_path is not None:
        outputs["source_manifest"] = _rel(source_manifest_path)

    manifest = {
        "version": 2,
        "figure_id": out_stem,
        "figure_kind": figure_kind,
        "pair_label": pair_label,
        "label_policy": label_policy
        or "left case A-H; right case I-P; source single-case letters stored separately",
        "cases": {
            "left": left_case,
            "right": right_case,
        },
        "outputs": outputs,
        "role_order": role_order,
        "panels": panels,
    }
    label_evidence = _label_evidence_summary_for_cases(
        [left_case, right_case],
        biovalidation_dir,
    )
    if label_evidence:
        manifest["label_evidence"] = label_evidence
    if parent_figure_id is not None:
        manifest["parent_figure_id"] = parent_figure_id
    if supplemental_outputs is not None:
        manifest["supplemental_outputs"] = supplemental_outputs
    if moved_to_supplement is not None:
        manifest["moved_to_supplement"] = moved_to_supplement
    path = biovalidation_dir / f"{out_stem}.label_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def _compose_main_pair(
    left_case: str,
    right_case: str,
    pair_label: str,
    biovalidation_dir: Path,
    out_dir: Path,
    out_stem: str,
    *,
    supplemental_outputs: dict[str, str],
) -> Path:
    """Build the concise manuscript-width pair figure."""
    img_l, manifest_l = _build_main_case_card(
        left_case,
        biovalidation_dir,
        side="left",
        figure_id=out_stem,
    )
    img_r, manifest_r = _build_main_case_card(
        right_case,
        biovalidation_dir,
        side="right",
        figure_id=out_stem,
    )

    canvas_w = CANVAS_MARGIN * 2 + CARD_W * 2 + CARD_GAP
    canvas_h = CANVAS_MARGIN + MAIN_CARD_H + CANVAS_MARGIN
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    left_x = CANVAS_MARGIN
    top_y = CANVAS_MARGIN
    right_x = CANVAS_MARGIN + CARD_W + CARD_GAP
    canvas.paste(img_l, (left_x, top_y))
    canvas.paste(img_r, (right_x, top_y))

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_stem}.png"
    pdf_path = out_dir / f"{out_stem}.pdf"

    canvas.save(png_path, dpi=(OUTPUT_DPI, OUTPUT_DPI))
    canvas.save(pdf_path, "PDF", resolution=float(OUTPUT_DPI))
    source_manifest_path = biovalidation_dir / f"{out_stem}.source_manifest.json"
    label_manifest_path = _write_label_manifest(
        out_stem=out_stem,
        pair_label=pair_label,
        left_case=left_case,
        right_case=right_case,
        panels=manifest_l + manifest_r,
        biovalidation_dir=biovalidation_dir,
        png_path=png_path,
        pdf_path=pdf_path,
        role_order=MAIN_PANEL_ROLE_ORDER,
        label_policy=(
            "main figure uses four narrative panels per case: "
            "left case A-D; right case E-H; dense detail panels are in "
            "the supplemental extended output"
        ),
        figure_kind="main",
        supplemental_outputs=supplemental_outputs,
        moved_to_supplement=SUPPLEMENT_DETAIL_ITEMS,
        source_manifest_path=source_manifest_path,
    )
    _write_source_manifest(
        out_stem=out_stem,
        pair_label=pair_label,
        left_case=left_case,
        right_case=right_case,
        panels=manifest_l + manifest_r,
        biovalidation_dir=biovalidation_dir,
        png_path=png_path,
        pdf_path=pdf_path,
        label_manifest_path=label_manifest_path,
        supplemental_outputs=supplemental_outputs,
    )
    log.info("wrote %s + .png", pdf_path)
    return pdf_path


def _compose_extended_pair(left_case: str, right_case: str, pair_label: str,
                           biovalidation_dir: Path, out_dir: Path, out_stem: str) -> Path:
    """Build a full-detail pair figure from two cached case-card panels."""
    img_l, manifest_l = _build_case_card(
        left_case,
        biovalidation_dir,
        side="left",
        figure_id=out_stem,
    )
    img_r, manifest_r = _build_case_card(
        right_case,
        biovalidation_dir,
        side="right",
        figure_id=out_stem,
    )

    canvas_w = CANVAS_MARGIN * 2 + CARD_W * 2 + CARD_GAP
    canvas_h = CANVAS_MARGIN + CARD_H + CANVAS_MARGIN
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    left_x = CANVAS_MARGIN
    top_y = CANVAS_MARGIN
    right_x = CANVAS_MARGIN + CARD_W + CARD_GAP
    canvas.paste(img_l, (left_x, top_y))
    canvas.paste(img_r, (right_x, top_y))

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_stem}.png"
    pdf_path = out_dir / f"{out_stem}.pdf"

    canvas.save(png_path, dpi=(OUTPUT_DPI, OUTPUT_DPI))
    canvas.save(pdf_path, "PDF", resolution=float(OUTPUT_DPI))
    source_manifest_path = biovalidation_dir / f"{out_stem}.source_manifest.json"
    label_manifest_path = _write_label_manifest(
        out_stem=out_stem,
        pair_label=pair_label,
        left_case=left_case,
        right_case=right_case,
        panels=manifest_l + manifest_r,
        biovalidation_dir=biovalidation_dir,
        png_path=png_path,
        pdf_path=pdf_path,
        role_order=PANEL_ROLE_ORDER,
        label_policy=(
            "extended detail output uses the full panel family: "
            "left case A-H; right case I-P; source single-case letters stored separately"
        ),
        figure_kind="extended_detail",
        parent_figure_id=out_stem.removesuffix("_extended"),
        source_manifest_path=source_manifest_path,
    )
    _write_source_manifest(
        out_stem=out_stem,
        pair_label=pair_label,
        left_case=left_case,
        right_case=right_case,
        panels=manifest_l + manifest_r,
        biovalidation_dir=biovalidation_dir,
        png_path=png_path,
        pdf_path=pdf_path,
        label_manifest_path=label_manifest_path,
    )
    log.info("wrote %s + .png", pdf_path)
    return pdf_path


def _compose_pair(left_case: str, right_case: str, pair_label: str,
                  biovalidation_dir: Path, out_dir: Path, out_stem: str) -> Path:
    """Build the single canonical paired-evidence figure for the manuscript."""
    return _compose_extended_pair(
        left_case,
        right_case,
        pair_label,
        biovalidation_dir,
        out_dir,
        out_stem,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default="all",
                   help="'all' or comma list (fig09,fig10,fig11).")
    p.add_argument("--biovalidation-dir", type=Path,
                   default=REPO_ROOT / "figures" / "biovalidation",
                   help="Where the per-case PNG/PDF inputs live.")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--allow-recompute", action="store_true",
                   help="Permit missing per-case artifacts to be rebuilt by running the compute pipeline.")
    p.add_argument("--rebuild-sidecars", action="store_true",
                   help="Recompute case payloads and export tabular sidecars used by paired redraw panels.")
    p.add_argument("--epochs", type=int, default=100,
                   help="Epochs for --rebuild-sidecars recomputation.")
    p.add_argument("--subsample-cells", type=int, default=3000,
                   help="Cell subsample for --rebuild-sidecars recomputation.")
    p.add_argument("--top-k-genes", type=int, default=5,
                   help="Top correlated genes per dim for --rebuild-sidecars recomputation.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.pairs == "all":
        targets = list(_PAIRS)
    else:
        targets = [t.strip() for t in args.pairs.split(",")]
        unknown = [t for t in targets if t not in _PAIRS]
        if unknown:
            raise SystemExit(f"unknown pair(s) {unknown}; choices: {list(_PAIRS)}")

    if args.rebuild_sidecars:
        unique_cases: list[str] = []
        for stem in targets:
            for case_id in _PAIRS[stem][:2]:
                if case_id not in unique_cases:
                    unique_cases.append(case_id)
        for case_id in unique_cases:
            log.info("[%s] rebuilding payload sidecar for direct G/H redraws", case_id)
            _rebuild_case_sidecar_and_artifact(
                case_id,
                args.biovalidation_dir,
                epochs=args.epochs,
                subsample_cells=args.subsample_cells,
                top_k_genes=args.top_k_genes,
            )

    n_done = n_err = 0
    for stem in targets:
        left, right, label, out_stem = _PAIRS[stem]
        log.info("=== %s · %s ===", stem, label)
        try:
            _ensure_case_artifacts(left,  args.biovalidation_dir,
                                   allow_recompute=args.allow_recompute)
            _ensure_case_artifacts(right, args.biovalidation_dir,
                                   allow_recompute=args.allow_recompute)
            _compose_pair(left, right, label,
                          args.biovalidation_dir, args.out_dir, out_stem)
            n_done += 1
        except Exception as exc:
            log.exception("[%s] failed: %s", stem, exc)
            n_err += 1
    log.info("biovalidation pairs done: %d new, %d errors", n_done, n_err)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
