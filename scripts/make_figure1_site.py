#!/usr/bin/env python3
"""Compose a paper-ready scCCVGBen site figure from browser screenshots.

Input  : figures/site_shots/{home,datasets,methods,metrics,dataset-detail}.png
Output : figures/fig1_scCCVGBen_site.{png,pdf}

The compositor intentionally avoids panel letters. It crops each screenshot to
the article content, trims excess white margins, and preserves screenshot aspect
ratios so typography is not stretched in the final paper figure. Capture the
source pages with a desktop viewport (for example 1800×3200, device scale 1)
and the site's xlarge text-scale option when regenerating the composed figure.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHOTS = ROOT / "figures" / "site_shots"
DEFAULT_OUT_DIR = ROOT / "figures"

DPI = 300
CARD_WIDTH = 1740
CARD_PAD = 38
TITLE_HEIGHT = 216
TITLE_FONT_PX = 78
CALLOUT_FONT_PX = 39
HEADER_HEIGHT = 250
HEADER_TITLE_FONT_PX = 82
HEADER_BODY_FONT_PX = 42
BADGE_FONT_PX = 38
GAP_X = 38
GAP_Y = 34
CANVAS_PAD = 30
MIN_DESKTOP_SHOT_WIDTH = 1400
MIN_RIGHT_SIDE_CONTENT_PIXELS = 500
# Three columns keep the cards aligned while balancing tall registry/detail
# screenshots so the final mosaic stays dense rather than leaving a sparse
# lower-right quadrant.
COLUMN_ORDER = ((0, 4, 7), (3, 2), (5, 6, 1))


@dataclass(frozen=True)
class Panel:
    image: str
    title: str
    callout: str
    crop: tuple[float, float, float, float]
    accent: str


PANELS = [
    # Home is 1600×3200 with sections: hero/KPI (0-0.15), composition (0.15-0.39),
    # model architecture (0.39-0.60), metadata distributions (0.60-0.94), explore (0.94+).
    Panel("home.png", "Benchmark overview",
            "Landing-page KPIs frame the 200-dataset benchmark before readers enter registries.",
            (0.190, 0.005, 1.000, 0.180), "#1f5f9f"),
    Panel("home.png", "Tissue and species composition",
            "Coverage summaries expose tissue, species, modality, and study-balance structure.",
            (0.190, 0.185, 1.000, 0.340), "#24989f"),
    Panel("home.png", "Metadata distributions (4 charts)",
            "Distribution panels make scale and metadata skew visible without leaving the overview.",
            (0.190, 0.575, 1.000, 0.875), "#5a7d2f"),
    Panel("datasets.png", "Dataset index with filters",
            "Searchable dataset cards connect GEO provenance, modality, organism, tissue, and task tags.",
            (0.190, 0.010, 1.000, 0.715), "#b7791f"),
    Panel("methods.png", "Method catalog (32 clickable cards)",
            "Method registry contrasts baseline families and scCCVGBen encoder/graph variants.",
            (0.190, 0.010, 1.000, 0.780), "#7a5ab8"),
    Panel("metrics.png", "Metric registry (26 with details)",
            "BEN, DRE, and LSE metric families are documented with definitions and interpretation notes.",
            (0.190, 0.010, 1.000, 0.380), "#b54848"),
    Panel("dataset-detail.png", "Per-dataset detail page",
            "Detail pages preserve dataset-level audit trails for capture, labels, and benchmark context.",
            (0.190, 0.010, 1.000, 0.520), "#24989f"),
    Panel("method-detail.png", "Per-method detail page",
            "Method pages summarize implementation choices so benchmark comparisons remain traceable.",
            (0.190, 0.010, 1.000, 0.500), "#7a5ab8"),
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


def _load(shots_dir: Path, name: str) -> Image.Image:
    path = shots_dir / name
    if not path.exists():
        raise FileNotFoundError(f"missing screenshot: {path}")
    img = Image.open(path).convert("RGB")
    _assert_desktop_capture(img, path)
    return img


def _assert_desktop_capture(img: Image.Image, path: Path) -> None:
    """Reject mobile/narrow captures before they make blank or truncated cards."""
    if img.width < MIN_DESKTOP_SHOT_WIDTH:
        raise ValueError(
            f"{path} is only {img.width}px wide; recapture with a desktop viewport "
            "(for example Chrome --window-size=1800,3200 --force-device-scale-factor=1)."
        )
    arr = np.asarray(img)
    # A mobile capture in a large screenshot leaves nearly all right-side pixels
    # as plain page background. Desktop captures have real article/card/chart
    # content beyond the side menu.
    right_region = arr[: int(img.height * 0.9), int(img.width * 0.35):, :]
    content_mask = np.any(right_region < 238, axis=2)
    if int(content_mask.sum()) < MIN_RIGHT_SIDE_CONTENT_PIXELS:
        raise ValueError(
            f"{path} looks like a mobile or horizontally clipped capture; "
            "recapture the page at desktop width with ?scale=xlarge."
        )


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


def _wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap text by rendered width so callouts remain readable on export."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current.append(word)
            continue
        if current:
            lines.append(" ".join(current))
        current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _draw_callout(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    fill: str = "#3b465c",
    line_gap: int = 8,
) -> int:
    x, y = xy
    height = 0
    for line in _wrapped_lines(draw, text, font, max_width):
        draw.text((x, y + height), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y + height), line, font=font)
        height += int(bbox[3] - bbox[1]) + line_gap
    return max(0, height - line_gap)


def _panel_card(panel: Panel, shots_dir: Path) -> Image.Image:
    img = _trim_white(_crop_fraction(_load(shots_dir, panel.image), panel.crop))
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
    _draw_callout(
        draw,
        (CARD_PAD, 112),
        panel.callout,
        font=_font(CALLOUT_FONT_PX),
        max_width=content_w,
    )
    card.paste(img, (CARD_PAD, TITLE_HEIGHT))
    return card


MIN_ASPECT_W_OVER_H = 17 / 21


def _column_height(column: list[Image.Image]) -> int:
    return sum(card.height for card in column) + GAP_Y * (len(column) - 1)


def _draw_header(draw: ImageDraw.ImageDraw, x: int, y: int, width: int) -> None:
    draw.rounded_rectangle(
        (x, y, x + width - 1, y + HEADER_HEIGHT - 1),
        radius=20,
        fill="#f3f7fb",
        outline="#d7e2ee",
        width=2,
    )
    draw.rounded_rectangle(
        (x, y, x + width - 1, y + 14),
        radius=20,
        fill="#1f5f9f",
        outline="#1f5f9f",
        width=1,
    )
    text_x = x + 42
    draw.text(
        (text_x, y + 36),
        "Interactive scCCVGBen benchmark atlas",
        font=_font(HEADER_TITLE_FONT_PX, bold=True),
        fill="#13213a",
    )
    body = (
        "A multi-page website mosaic links dataset provenance, method registry, "
        "metric families, and drill-down audit trails used by the benchmark."
    )
    for line_idx, line in enumerate(wrap(body, width=105)):
        draw.text(
            (text_x, y + 132 + line_idx * 50),
            line,
            font=_font(HEADER_BODY_FONT_PX),
            fill="#3b465c",
        )

    badges = [
        ("200 datasets", "#1f5f9f"),
        ("32 methods", "#7a5ab8"),
        ("26 metrics", "#b54848"),
        ("detail pages", "#24989f"),
    ]
    badge_widths = [
        draw.textbbox((0, 0), label, font=_font(BADGE_FONT_PX, bold=True))[2] + 42
        for label, _ in badges
    ]
    badge_x = x + width - sum(badge_widths) - 20 * (len(badges) - 1) - 42
    badge_y = y + HEADER_HEIGHT - 72
    for (label, color), badge_w in zip(badges, badge_widths, strict=True):
        draw.rounded_rectangle(
            (badge_x, badge_y, badge_x + badge_w, badge_y + 48),
            radius=24,
            fill=color,
            outline=color,
            width=1,
        )
        draw.text(
            (badge_x + 21, badge_y + 3),
            label,
            font=_font(BADGE_FONT_PX, bold=True),
            fill="white",
        )
        badge_x += badge_w + 20


def _build_canvas(shots_dir: Path) -> Image.Image:
    cards = [_panel_card(panel, shots_dir) for panel in PANELS]
    columns = [[cards[i] for i in order] for order in COLUMN_ORDER]
    col_heights = [_column_height(column) for column in columns]
    target_col_height = max(col_heights)

    n_cols = len(columns)
    mosaic_width = CARD_WIDTH * n_cols + GAP_X * (n_cols - 1)
    content_width = CANVAS_PAD * 2 + mosaic_width
    content_height = CANVAS_PAD * 2 + HEADER_HEIGHT + GAP_Y + target_col_height
    required_w = int(content_height * MIN_ASPECT_W_OVER_H) + 1
    width = max(content_width, required_w)
    height = content_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    x_start = CANVAS_PAD
    extra = width - content_width
    if extra > 0:
        x_start = CANVAS_PAD + extra // 2

    _draw_header(draw, x_start, CANVAS_PAD, mosaic_width)

    for col_idx, column in enumerate(columns):
        x = x_start + col_idx * (CARD_WIDTH + GAP_X)
        y = CANVAS_PAD + HEADER_HEIGHT + GAP_Y
        for card_idx, card in enumerate(column):
            canvas.paste(card, (x, y))
            y += card.height
            if card_idx < len(column) - 1:
                y += GAP_Y

    return canvas


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots-dir",
        type=Path,
        default=DEFAULT_SHOTS,
        help="Directory containing browser screenshots (default: figures/site_shots).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for rendered PNG/PDF outputs (default: figures).",
    )
    parser.add_argument(
        "--partial-ok",
        action="store_true",
        help="Accepted for orchestrator parity; this static composition is all-or-fail.",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=0,
        help="Accepted for orchestrator parity; not used by this static composition.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "fig1_scCCVGBen_site.png"
    out_pdf = out_dir / "fig1_scCCVGBen_site.pdf"

    canvas = _build_canvas(args.shots_dir)
    width, height = canvas.size
    canvas.save(out_png, dpi=(DPI, DPI))
    canvas.save(out_pdf, "PDF", resolution=DPI)
    print(f"wrote {out_png}  ({width} × {height}, W/H={width/height:.3f}, target > {MIN_ASPECT_W_OVER_H:.3f})")
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
