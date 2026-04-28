#!/usr/bin/env python3
"""Compose a paper-ready scCCVGBen site figure from browser screenshots.

Input  : figures/site_shots/{home,datasets,methods,metrics,dataset-detail}.png
Output : figures/fig1_scCCVGBen_site.{png,pdf}

The compositor adds explicit A-H panel labels in visual reading order, crops
each screenshot to the article content, trims excess white margins, and lays
the panels out in three wide columns so the captured page content remains
legible after the figure is scaled to manuscript text width. Capture the source
pages with a desktop viewport (for example 1800×4600, device scale 1) and the
site's xlarge text-scale option when regenerating the composed figure.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scccvgben.figures.fonts import arial_font_path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHOTS = ROOT / "figures" / "site_shots"
DEFAULT_OUT_DIR = ROOT / "figures"

DPI = 300
CARD_WIDTH = 1900
CARD_PAD = 28
TITLE_HEIGHT = 134
TITLE_FONT_PX = 78
HEADER_HEIGHT = 170
HEADER_TITLE_FONT_PX = 92
BADGE_FONT_PX = 60
BADGE_GAP_PX = 30
GAP_X = 28
GAP_Y = 26
CANVAS_PAD = 22
MIN_DESKTOP_SHOT_WIDTH = 1400
MIN_RIGHT_SIDE_CONTENT_PIXELS = 500
# Three visual columns keep each screenshot larger after the composed figure is
# scaled to manuscript text width.  Row-major labels match the caption:
# A-C homepage sections, D dataset index, E dataset detail, F method catalogue,
# G method detail, H metric taxonomy.
COLUMN_ORDER = ((0, 3, 7), (1, 6, 5), (2, 4))


@dataclass(frozen=True)
class Panel:
    image: str
    title: str
    crop: tuple[float, float, float, float]
    accent: str


PANELS = [
    # Home captures use a tall 1800×4600 viewport so source-page typography can stay
    # large while the composition/metadata panels are cropped to their own sections.
    Panel("home.png", "Atlas entry point",
          (0.245, 0.000, 1.000, 0.176), "#1f5f9f"),
    Panel("home.png", "Cohort balance view",
          (0.245, 0.180, 1.000, 0.330), "#24989f"),
    Panel("home.png", "Scale audit charts",
          (0.245, 0.428, 1.000, 0.673), "#5a7d2f"),
    Panel("datasets.png", "Dataset browser",
          (0.240, 0.010, 1.000, 0.195), "#b7791f"),
    Panel("methods.png", "Comparator registry",
          (0.240, 0.010, 1.000, 0.345), "#7a5ab8"),
    Panel("metrics.png", "Metric taxonomy",
          (0.240, 0.010, 1.000, 0.340), "#b54848"),
    Panel("dataset-detail.png", "Dataset audit trail",
          (0.240, 0.010, 1.000, 0.270), "#24989f"),
    Panel("method-detail.png", "Method audit trail",
          (0.240, 0.010, 1.000, 0.255), "#7a5ab8"),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    arial = arial_font_path(bold=bold)
    if arial is not None:
        return ImageFont.truetype(str(arial), size=size)
    names = [
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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


def _panel_card(panel: Panel, shots_dir: Path, label: str) -> Image.Image:
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
    label_box = (CARD_PAD, 24, CARD_PAD + 90, 114)
    draw.rounded_rectangle(
        label_box,
        radius=12,
        fill=panel.accent,
        outline=panel.accent,
        width=1,
    )
    label_font = _font(76, bold=True)
    lb = draw.textbbox((0, 0), label, font=label_font)
    draw.text(
        (
            label_box[0] + (label_box[2] - label_box[0] - (lb[2] - lb[0])) // 2,
            label_box[1] + (label_box[3] - label_box[1] - (lb[3] - lb[1])) // 2 - 5,
        ),
        label,
        font=label_font,
        fill="white",
    )
    draw.text(
        (CARD_PAD + 118, 18),
        panel.title,
        font=_font(TITLE_FONT_PX, bold=True),
        fill="#172033",
    )
    # Per-panel callout prose is omitted: the captured screenshot already
    # carries an in-page H1 / lede so a second sentence above the image only
    # duplicates information. Text moved to the LaTeX figure caption.
    card.paste(img, (CARD_PAD, TITLE_HEIGHT))
    return card


MIN_ASPECT_W_OVER_H = 17 / 21


def _column_height(column: list[Image.Image]) -> int:
    return sum(card.height for card in column) + GAP_Y * (len(column) - 1)


def _visual_labels(column_order: tuple[tuple[int, ...], ...]) -> dict[int, str]:
    """Assign panel letters by row-major reading order rather than source list order."""
    labels: dict[int, str] = {}
    next_label = ord("A")
    for row_idx in range(max(len(column) for column in column_order)):
        for column in column_order:
            if row_idx >= len(column):
                continue
            panel_idx = column[row_idx]
            labels[panel_idx] = chr(next_label)
            next_label += 1

    expected = set(range(len(PANELS)))
    if set(labels) != expected:
        missing = sorted(expected - set(labels))
        extra = sorted(set(labels) - expected)
        raise ValueError(
            f"Figure 1 layout does not cover panels exactly once; missing={missing}, extra={extra}"
        )
    return labels


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
    title_y = y + (HEADER_HEIGHT - HEADER_TITLE_FONT_PX) // 2 - 6
    draw.text(
        (text_x, title_y),
        "Interactive scCCVGBen benchmark atlas",
        font=_font(HEADER_TITLE_FONT_PX, bold=True),
        fill="#13213a",
    )
    # Long descriptive subtitle moved to the LaTeX figure caption per
    # paper-layout policy. The header now carries only the title plus
    # the colored count badges below.

    badges = [
        ("200 datasets", "#1f5f9f"),
        ("32 methods", "#7a5ab8"),
        ("20 display metrics", "#b54848"),
        ("detail pages", "#24989f"),
    ]
    badge_widths = [
        draw.textbbox((0, 0), label, font=_font(BADGE_FONT_PX, bold=True))[2] + 48
        for label, _ in badges
    ]
    badge_x = x + width - sum(badge_widths) - BADGE_GAP_PX * (len(badges) - 1) - 42
    # Badge row vertically centered against the title baseline; pulls in
    # close to the title now that the header is shorter.
    badge_y = y + (HEADER_HEIGHT - 70) // 2
    for (label, color), badge_w in zip(badges, badge_widths, strict=True):
        draw.rounded_rectangle(
            (badge_x, badge_y, badge_x + badge_w, badge_y + 58),
            radius=29,
            fill=color,
            outline=color,
            width=1,
        )
        draw.text(
            (badge_x + 24, badge_y + 4),
            label,
            font=_font(BADGE_FONT_PX, bold=True),
            fill="white",
        )
        badge_x += badge_w + BADGE_GAP_PX


def _build_canvas(shots_dir: Path) -> Image.Image:
    labels = _visual_labels(COLUMN_ORDER)
    cards = [
        _panel_card(panel, shots_dir, labels[idx])
        for idx, panel in enumerate(PANELS)
    ]
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
