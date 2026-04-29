#!/usr/bin/env python3
"""Compose the canonical manuscript Figure 2.

Stacks the dataset-metadata overview above the scCCVGBen architecture
diagram into the single manuscript-facing output stem
``figures/fig02_benchmark_architecture.{pdf,png}``. The component images are
intermediates and are purged after the composite is written so the final stem
remains the only canonical Figure 2 output in ``figures/``.

Inputs (rendered on-demand if missing — this script is self-contained):
- figures/fig01_dataset_metadata.png — dataset metadata overview (panels A-C)
- figures/fig02_architecture.png     — architecture-only panel D intermediate

The composite adds no figure-number title — the manuscript assigns the
"Figure 2" caption.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scccvgben.figures.fonts import arial_font_path  # noqa: E402

log = logging.getLogger(__name__)

DPI = 300
TARGET_WIDTH = 4400
GAP = 10
PAD = 10


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    arial = arial_font_path(bold=bold)
    if arial is not None:
        return ImageFont.truetype(str(arial), size=size)
    fallbacks = [
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in fallbacks:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _scale_to_width(img: Image.Image, target_w: int) -> Image.Image:
    if img.width == target_w:
        return img
    new_h = max(1, round(img.height * target_w / img.width))
    return img.resize((target_w, new_h), Image.Resampling.LANCZOS)


def _trim_whitespace(img: Image.Image, *, tolerance: int = 248, pad: int = 10) -> Image.Image:
    gray = img.convert("L")
    mask = gray.point(lambda value: 255 if value < tolerance else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(img.width, x1 + pad)
    y1 = min(img.height, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0, y0, x1, y1))


def _ensure_top(top: Path, out_dir: Path) -> None:
    """Render figures/fig01_dataset_metadata.png on-demand if missing."""
    if top.exists():
        return
    log.info("missing top image %s — rendering via make_fig01_dataset_metadata", top)
    from scripts.make_fig01_dataset_metadata import main as fig01_main
    fig01_main(["--out-dir", str(out_dir), "--partial-ok"])


def _ensure_bottom(bottom: Path, out_dir: Path) -> None:
    """Render the architecture panel on-demand if it's not on disk."""
    if bottom.exists():
        return
    log.info(
        "missing bottom image %s — rendering via make_figure2_model_architecture",
        bottom,
    )
    from scripts.make_figure2_model_architecture import make_figure

    make_figure(out_dir, site_static=None, stem=bottom.stem)
    if not bottom.exists():
        raise FileNotFoundError(f"expected rendered intermediate {bottom}")


_INTERMEDIATE_STEMS = ("fig01_dataset_metadata", "fig02_architecture")


def _purge_intermediates(out_dir: Path) -> None:
    """Delete metadata + architecture inputs after the canonical composite is written."""
    for stem in _INTERMEDIATE_STEMS:
        for ext in ("png", "pdf"):
            p = out_dir / f"{stem}.{ext}"
            if p.exists():
                p.unlink()
                log.info("removed intermediate %s", p)


def build(top: Path, bottom: Path, out_dir: Path, *, stem: str = "fig02_benchmark_architecture") -> list[Path]:
    _ensure_top(top, out_dir)
    _ensure_bottom(bottom, out_dir)

    a = _scale_to_width(_trim_whitespace(Image.open(top).convert("RGB")), TARGET_WIDTH)
    b = _scale_to_width(_trim_whitespace(Image.open(bottom).convert("RGB")), TARGET_WIDTH)

    width = TARGET_WIDTH + PAD * 2
    height = PAD + a.height + GAP + b.height + PAD
    canvas = Image.new("RGB", (width, height), "white")

    y = PAD
    canvas.paste(a, (PAD, y))
    y += a.height + GAP
    canvas.paste(b, (PAD, y))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    canvas.save(out_png, dpi=(DPI, DPI))
    canvas.save(out_pdf, "PDF", resolution=DPI)

    # Only the final composite needs to live in figures/; the metadata and
    # architecture inputs are intermediates rebuilt on demand by the
    # _ensure_* fallbacks.
    _purge_intermediates(out_dir)
    return [out_png, out_pdf]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--top", type=Path,
                        default=REPO_ROOT / "figures" / "fig01_dataset_metadata.png")
    parser.add_argument("--bottom", type=Path,
                        default=REPO_ROOT / "figures" / "fig02_architecture.png")
    parser.add_argument("--out", default="fig02_benchmark_architecture",
                        help="Output stem (default fig02_benchmark_architecture).")
    parser.add_argument("--partial-ok", action="store_true",
                        help="Accepted for orchestrator parity; this is all-or-fail.")
    parser.add_argument("--target-n", type=int, default=0,
                        help="Accepted for orchestrator parity; not used.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    outputs = build(args.top, args.bottom, args.out_dir, stem=args.out)
    for path in outputs:
        log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
