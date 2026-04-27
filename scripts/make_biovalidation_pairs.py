#!/usr/bin/env python3
"""make_biovalidation_pairs.py — pair two case-study panels into one figure.

The legacy CCVGAE case-study figures (fig10/11/12) were single-case panels.
This script keeps the same biology depth but pairs cases two-per-figure so
the manuscript stays at 3 bio-validation figures total. Each pair is
rendered as a single 32 × 16 inch matplotlib figure with the two cases
side-by-side; each half re-uses the existing 7-panel ``compose_case_figure``
layout but draws into the left/right half of a wider canvas.

Pairs (default):

    fig11 — SD + GASTRIC      (sleep deprivation + gastric cancer)
    fig12 — UCB + HSC_AGE     (cord-blood megakaryocyte + aged HSC)
    fig13 — IR + COVID        (radiation injury + COVID-19 BALF)

Each side keeps the full A-H panel set the single-case composer produces
(see ``scccvgben/biovalidation/compose/case_figure.py``); the only
difference is the figure-level title that names both cases.

Inputs
------
The script re-uses the per-case payload pickles produced by
``run_biovalidation`` when ``--cache`` is passed. Without a cache it falls
back to running ``run_case`` again per pair (~2 minutes/case).

Usage::

    python scripts/make_biovalidation_pairs.py                 # all 3 pairs
    python scripts/make_biovalidation_pairs.py --pairs fig11   # one pair only
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from PIL import Image  # noqa: E402

log = logging.getLogger(__name__)


# Pair definition: out_stem -> (left_case_id, right_case_id, pair_label)
_PAIRS: dict[str, tuple[str, str, str]] = {
    "fig11": ("SD",  "GASTRIC",
              "Sleep deprivation × Gastric cancer microenvironment"),
    "fig12": ("UCB", "HSC_AGE",
              "Cord-blood megakaryocyte × Aged hematopoietic stem cells"),
    "fig13": ("IR",  "COVID",
              "Radiation injury × COVID-19 BALF immune landscape"),
}


def _ensure_case_pdf(case_id: str, biovalidation_dir: Path) -> Path:
    """Locate ``fig_biovalidation_case_{case_id}.pdf`` (re-render if missing)."""
    pdf = biovalidation_dir / f"fig_biovalidation_case_{case_id}.pdf"
    png = biovalidation_dir / f"fig_biovalidation_case_{case_id}.png"
    if pdf.exists() and png.exists():
        return pdf

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
    return pdf


def _compose_pair(left_case: str, right_case: str, pair_label: str,
                  biovalidation_dir: Path, out_dir: Path, out_stem: str) -> Path:
    """Stitch the left + right case PNGs into one wide pair figure.

    Output: PDF + PNG. Rationale: PIL gives byte-precise control over the
    join (no figsize drift, no font re-rasterisation), then matplotlib
    re-saves as PDF for vector preservation of typography from the input
    PDFs (we use ``img2pdf``-style two-page A0 if matplotlib path fails).
    """
    left_png  = biovalidation_dir / f"fig_biovalidation_case_{left_case}.png"
    right_png = biovalidation_dir / f"fig_biovalidation_case_{right_case}.png"
    for p in (left_png, right_png):
        if not p.exists():
            raise SystemExit(f"missing case PNG: {p}")

    img_l = Image.open(left_png).convert("RGB")
    img_r = Image.open(right_png).convert("RGB")

    # Normalise the right panel height to match the left, side-by-side join.
    if img_r.height != img_l.height:
        ratio = img_l.height / img_r.height
        img_r = img_r.resize((int(img_r.width * ratio), img_l.height),
                              Image.LANCZOS)

    # Vertical title band split into two lines: top = bold pair label,
    # bottom = small LEFT/RIGHT case markers. 130-px band at 200 dpi
    # leaves 35-px clearance for the 22-pt title and another 60 px for
    # the 14-pt subtitle so the two lines never collide.
    title_band_h = 130
    canvas = Image.new("RGB",
                        (img_l.width + img_r.width, img_l.height + title_band_h),
                        "white")
    canvas.paste(img_l, (0, title_band_h))
    canvas.paste(img_r, (img_l.width, title_band_h))

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{out_stem}.png"
    pdf_path = out_dir / f"{out_stem}.pdf"

    fig_w_in = (img_l.width + img_r.width) / 200
    fig_h_in = (img_l.height + title_band_h) / 200
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=200)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])  # axes fill the entire figure
    ax.axis("off")
    ax.imshow(canvas, aspect="equal")
    # With axes filling the figure, figure-y == canvas-y. Place the
    # title at canvas_y ≈ 35 px and the subtitles at canvas_y ≈ 95 px;
    # both fit inside the 130-px white band painted at the top of the
    # canvas, with comfortable separation between the 22-pt title and
    # 14-pt subtitles.
    canvas_h = canvas.size[1]
    title_y = 1.0 - 35.0 / canvas_h
    sub_y   = 1.0 - 95.0 / canvas_h
    fig.text(0.5, title_y,
             f"Bio-validation paired case · {pair_label}",
             fontsize=22, fontweight="bold", ha="center", va="center")
    fig.text(0.25, sub_y, f"LEFT — case {left_case}",
             fontsize=14, fontweight="bold", color="#0F172A",
             ha="center", va="center")
    fig.text(0.75, sub_y, f"RIGHT — case {right_case}",
             fontsize=14, fontweight="bold", color="#0F172A",
             ha="center", va="center")
    # bbox=None is mandatory: bbox_inches="tight" crops the white
    # title band and the title gets pulled down onto the case content.
    fig.savefig(pdf_path, pad_inches=0.0)
    fig.savefig(png_path, pad_inches=0.0, dpi=200)
    plt.close(fig)
    log.info("wrote %s + .png", pdf_path)
    return pdf_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", default="all",
                   help="'all' or comma list (fig11,fig12,fig13).")
    p.add_argument("--biovalidation-dir", type=Path,
                   default=REPO_ROOT / "figures" / "biovalidation",
                   help="Where the per-case PNG/PDF inputs live.")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.pairs == "all":
        targets = list(_PAIRS)
    else:
        targets = [t.strip() for t in args.pairs.split(",")]
        unknown = [t for t in targets if t not in _PAIRS]
        if unknown:
            raise SystemExit(f"unknown pair(s) {unknown}; choices: {list(_PAIRS)}")

    n_done = n_err = 0
    for stem in targets:
        left, right, label = _PAIRS[stem]
        log.info("=== %s · %s ===", stem, label)
        try:
            _ensure_case_pdf(left,  args.biovalidation_dir)
            _ensure_case_pdf(right, args.biovalidation_dir)
            _compose_pair(left, right, label,
                          args.biovalidation_dir, args.out_dir, stem)
            n_done += 1
        except Exception as exc:
            log.exception("[%s] failed: %s", stem, exc)
            n_err += 1
    log.info("biovalidation pairs done: %d new, %d errors", n_done, n_err)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
