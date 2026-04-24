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
CARD_PAD = 24
TITLE_HEIGHT = 58
GAP_X = 48
GAP_Y = 42
CANVAS_PAD = 34
COLUMN_ORDER = ((0, 2, 5), (1, 3, 4))


@dataclass(frozen=True)
class Panel:
    image: str
    title: str
    crop: tuple[float, float, float, float]
    accent: str


PANELS = [
    Panel("home.png", "Benchmark overview", (0.175, 0.015, 0.985, 0.235), "#1f5f9f"),
    Panel("home.png", "Tissue and species composition", (0.175, 0.235, 0.985, 0.445), "#24989f"),
    Panel("datasets.png", "Dataset index", (0.175, 0.015, 0.985, 0.705), "#b7791f"),
    Panel("methods.png", "Method catalog", (0.175, 0.015, 0.985, 0.775), "#7a5ab8"),
    Panel("metrics.png", "Metric registry", (0.175, 0.015, 0.985, 0.275), "#b54848"),
    Panel("dataset-detail.png", "Dataset detail page", (0.175, 0.015, 0.985, 0.425), "#24989f"),
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


def _trim_white(img: Image.Image, pad: int = 26, threshold: int = 236) -> Image.Image:
    arr = np.asarray(img)
    mask = np.any(arr < threshold, axis=2)
    if not mask.any():
        return img
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
        (CARD_PAD, 17),
        panel.title,
        font=_font(30, bold=True),
        fill="#172033",
    )
    card.paste(img, (CARD_PAD, TITLE_HEIGHT))
    return card


def main() -> None:
    cards = [_panel_card(panel) for panel in PANELS]
    columns = [[cards[i] for i in order] for order in COLUMN_ORDER]
    col_heights = [
        sum(card.height for card in column) + GAP_Y * (len(column) - 1)
        for column in columns
    ]

    width = CANVAS_PAD * 2 + CARD_WIDTH * 2 + GAP_X
    height = CANVAS_PAD * 2 + max(col_heights)
    canvas = Image.new("RGB", (width, height), "white")

    for col_idx, column in enumerate(columns):
        x = CANVAS_PAD + col_idx * (CARD_WIDTH + GAP_X)
        y = CANVAS_PAD
        for card in column:
            canvas.paste(card, (x, y))
            y += card.height + GAP_Y

    canvas.save(OUT_PNG, dpi=(DPI, DPI))
    canvas.save(OUT_PDF, "PDF", resolution=DPI)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
