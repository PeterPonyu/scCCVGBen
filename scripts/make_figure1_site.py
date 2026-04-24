#!/usr/bin/env python3
"""Compose a paper-ready scCCVGBen site figure from browser screenshots.

Input  : figures/site_shots/{home,datasets,methods,metrics,dataset-detail}.png
Output : figures/fig1_scCCVGBen_site.{png,pdf}

The compositor intentionally avoids panel letters. It crops each screenshot to
the article content, trims excess white margins, and preserves screenshot aspect
ratios so typography is not stretched in the final paper figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "figures" / "site_shots"
OUT_PNG = ROOT / "figures" / "fig1_scCCVGBen_site.png"
OUT_PDF = ROOT / "figures" / "fig1_scCCVGBen_site.pdf"

DPI = 300
CARD_WIDTH = 1600
CARD_PAD = 30
TITLE_HEIGHT = 84
TITLE_FONT_PX = 44
GAP_X = 54
GAP_Y = 48
CANVAS_PAD = 40
# Three columns keep the cards aligned without large bottom gaps.
COLUMN_ORDER = ((0, 1, 6, 7), (4, 5), (2, 3))


@dataclass(frozen=True)
class Panel:
    image: str
    title: str
    crop: tuple[float, float, float, float]
    accent: str


PANELS = [
    # Home is 1600×3200 with sections: hero/KPI (0-0.15), composition (0.15-0.27),
    # metadata distributions (0.27-0.48), explore (0.48+).
    Panel("home.png", "Benchmark overview",
          (0.165, 0.005, 0.995, 0.122), "#1f5f9f"),
    Panel("home.png", "Tissue and species composition",
          (0.165, 0.145, 0.995, 0.245), "#24989f"),
    Panel("home.png", "Metadata distributions (4 charts)",
          (0.165, 0.275, 0.995, 0.505), "#5a7d2f"),
    Panel("datasets.png", "Dataset index with filters",
          (0.165, 0.010, 0.995, 0.715), "#b7791f"),
    Panel("methods.png", "Method catalog (32 clickable cards)",
          (0.165, 0.010, 0.995, 0.780), "#7a5ab8"),
    Panel("metrics.png", "Metric registry (26 with details)",
          (0.165, 0.010, 0.995, 0.380), "#b54848"),
    Panel("dataset-detail.png", "Per-dataset detail page",
          (0.165, 0.010, 0.995, 0.520), "#24989f"),
    Panel("method-detail.png", "Per-method detail page",
          (0.165, 0.010, 0.995, 0.500), "#7a5ab8"),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load(name: str) -> Image.Image:
    path = SHOTS / name
    if not path.exists():
        raise FileNotFoundError(f"missing screenshot: {path}")
    return Image.open(path).convert("RGB")


def _crop_fraction(img: Image.Image, crop: tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    left, top, right, bottom = crop
    box = (
        max(0, round(w * left)),
        max(0, round(h * top)),
        min(w, round(w * right)),
        min(h, round(h * bottom)),
    )
    return img.crop(box)


def _trim_white(
    img: Image.Image,
    pad: int = 26,
    threshold: int = 236,
    min_density: float = 0.006,
) -> Image.Image:
    arr = np.asarray(img)
    mask = np.any(arr < threshold, axis=2)
    if not mask.any():
        return img
    row_min = max(3, round(img.width * min_density))
    col_min = max(3, round(img.height * min_density))
    ys = np.where(mask.sum(axis=1) >= row_min)[0]
    xs = np.where(mask.sum(axis=0) >= col_min)[0]
    if len(ys) == 0 or len(xs) == 0:
        ys, xs = np.where(mask)
    left = max(0, int(xs.min()) - pad)
    top = max(0, int(ys.min()) - pad)
    right = min(img.width, int(xs.max()) + pad + 1)
    bottom = min(img.height, int(ys.max()) + pad + 1)
    return img.crop((left, top, right, bottom))


def _panel_card(panel: Panel) -> Image.Image:
    img = _trim_white(_crop_fraction(_load(panel.image), panel.crop))
    content_w = CARD_WIDTH - CARD_PAD * 2
    content_h = max(1, round(img.height * content_w / img.width))
    img = img.resize((content_w, content_h), Image.Resampling.LANCZOS)

    card_h = TITLE_HEIGHT + content_h + CARD_PAD
    card = Image.new("RGB", (CARD_WIDTH, card_h), "white")
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle(
        (0, 0, CARD_WIDTH - 1, card_h - 1),
        radius=8,
        fill="white",
        outline="#d8dee8",
        width=2,
    )
    draw.rounded_rectangle(
        (0, 0, CARD_WIDTH - 1, 8),
        radius=8,
        fill=panel.accent,
        outline=panel.accent,
        width=1,
    )
    draw.text(
        (CARD_PAD, 24),
        panel.title,
        font=_font(TITLE_FONT_PX, bold=True),
        fill="#172033",
    )
    card.paste(img, (CARD_PAD, TITLE_HEIGHT))
    return card


MIN_ASPECT_W_OVER_H = 17 / 21


def _column_height(column: list[Image.Image]) -> int:
    return sum(card.height for card in column) + GAP_Y * (len(column) - 1)


def _column_gaps(column_height: int, target_height: int, gap_count: int) -> list[int]:
    if gap_count <= 0:
        return []
    extra = max(0, target_height - column_height)
    extra_each, remainder = divmod(extra, gap_count)
    return [
        GAP_Y + extra_each + (1 if idx < remainder else 0)
        for idx in range(gap_count)
    ]


def main() -> None:
    cards = [_panel_card(panel) for panel in PANELS]
    columns = [[cards[i] for i in order] for order in COLUMN_ORDER]
    col_heights = [_column_height(column) for column in columns]
    target_col_height = max(col_heights)

    n_cols = len(columns)
    content_width = CANVAS_PAD * 2 + CARD_WIDTH * n_cols + GAP_X * (n_cols - 1)
    content_height = CANVAS_PAD * 2 + target_col_height
    required_w = int(content_height * MIN_ASPECT_W_OVER_H) + 1
    width = max(content_width, required_w)
    height = content_height
    canvas = Image.new("RGB", (width, height), "white")

    x_start = CANVAS_PAD
    extra = width - content_width
    if extra > 0:
        x_start = CANVAS_PAD + extra // 2

    for col_idx, column in enumerate(columns):
        x = x_start + col_idx * (CARD_WIDTH + GAP_X)
        y = CANVAS_PAD
        gaps = _column_gaps(col_heights[col_idx], target_col_height, len(column) - 1)
        for card_idx, card in enumerate(column):
            canvas.paste(card, (x, y))
            y += card.height
            if card_idx < len(gaps):
                y += gaps[card_idx]

    canvas.save(OUT_PNG, dpi=(DPI, DPI))
    canvas.save(OUT_PDF, "PDF", resolution=DPI)
    print(f"wrote {OUT_PNG}  ({width} × {height}, W/H={width/height:.3f}, target > {MIN_ASPECT_W_OVER_H:.3f})")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
