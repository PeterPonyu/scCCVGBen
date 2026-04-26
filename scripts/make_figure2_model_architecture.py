#!/usr/bin/env python3
"""Generate Figure 2: wide scCCVGBen model architecture and benchmark axes.

The diagram is model-architecture-first and intentionally separates the four
paper-facing axes requested by the project spec:
  1. graph construction choices,
  2. encoder plugin registry,
  3. dual reconstruction paths,
  4. metric evaluation output.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scccvgben.figures.fonts import register_arial_with_matplotlib

log = logging.getLogger(__name__)
register_arial_with_matplotlib()

# Match Figure 1's clean publication palette while using architecture-specific rails.
C_ORANGE = "#D35400"
C_BLUE = "#2471A3"
C_TEAL = "#138D75"
C_PURPLE = "#6C3483"
C_GREEN = "#229954"
C_DARK = "#1F2D3D"
C_MUTED = "#64748B"
C_BG = "#F8FAFC"
C_CARD = "#FFFFFF"
C_LINE = "#CBD5E1"

ENCODERS = [
    "GAT", "GATv2", "Trans-\nformer", "Super\nGAT",
    "GCN", "SAGE", "Graph", "Cheb", "TAG", "ARMA", "SG", "SSG",
    "GIN", "Edge\nConv",
]
GRAPH_METHODS = ["kNN-Euclid", "kNN-cosine", "SNN", "Mutual kNN", "Gaussian threshold"]
BEN_METRICS = ["ASW", "DAV", "CAL"]
DRE_METRICS = ["DC", "QL", "QG", "Kmax", "Overall"]
LSE_METRICS = ["MD", "SDR", "PR", "AS", "TD", "NR", "Overall"]

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "font.size": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_W = 20.2
FIG_H = 8.45
AX_XMAX = 20.35
AX_YMIN = 0.12
AX_YMAX = 7.95


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str,
         *, fc: str = C_CARD, ec: str = C_LINE, lw: float = 1.2,
         radius: float = 0.12, color: str = C_DARK, size: float = 9,
         weight: str = "normal", ha: str = "center") -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    text_x = {"left": x + 0.16, "right": x + w - 0.16}.get(ha, x + w / 2)
    ax.text(
        text_x,
        y + h / 2,
        label,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        fontweight=weight,
        linespacing=1.05 if "\n" in label else 1.2,
        zorder=4.0,
    )
    return patch


def _label(ax: plt.Axes, x: float, y: float, text: str, *, size: float = 10.6,
           color: str = C_DARK, weight: str = "bold", ha: str = "left") -> None:
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
            fontweight=weight, zorder=4)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
           *, color: str = C_MUTED, lw: float = 1.8, rad: float = 0.0,
           zorder: float = 1.85, mutation_scale: float = 15.0) -> None:
    arr = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arr)


def _path_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str = C_MUTED,
    lw: float = 1.8,
    zorder: float = 1.85,
    mutation_scale: float = 15.0,
) -> None:
    """Draw a routed connector so arrows use whitespace corridors instead of labels."""
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    arr = FancyArrowPatch(
        path=MplPath(points, codes),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        zorder=zorder,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(arr)


def _flow_label(ax: plt.Axes, x: float, y: float, text: str, *, color: str) -> None:
    ax.text(
        x,
        y,
        text,
        fontsize=9.4,
        color=color,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.12,rounding_size=0.05",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
        zorder=4.5,
    )


def _pill_row(ax: plt.Axes, labels: Iterable[str], x: float, y: float, w: float,
              h: float, *, cols: int, color: str, size: float = 8.8) -> None:
    labels = list(labels)
    gap_x = 0.08
    gap_y = 0.08
    cell_w = (w - gap_x * (cols - 1)) / cols
    rows = (len(labels) + cols - 1) // cols
    cell_h = (h - gap_y * (rows - 1)) / rows
    for i, lab in enumerate(labels):
        row = i // cols
        col = i % cols
        xx = x + col * (cell_w + gap_x)
        yy = y + h - (row + 1) * cell_h - row * gap_y
        _box(ax, xx, yy, cell_w, cell_h, lab, fc="white", ec=color,
             lw=0.9, radius=0.08, color=C_DARK, size=size, weight="bold")


def _wrapped(text: str, width: int = 42) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _metric_family_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    family: str,
    title: str,
    body: str,
    *,
    size: float = 9.2,
) -> None:
    """Draw one grouped evaluation-family card without expanding it into a catalogue."""
    _box(ax, x, y, w, h, "", fc="white", ec=C_PURPLE, lw=1.1, radius=0.10)
    ax.text(
        x + 0.16,
        y + h - 0.22,
        family,
        ha="left",
        va="top",
        fontsize=size + 0.2,
        fontweight="bold",
        color=C_PURPLE,
        zorder=4,
    )
    ax.text(
        x + 0.72,
        y + h - 0.22,
        title,
        ha="left",
        va="top",
        fontsize=size,
        fontweight="bold",
        color=C_DARK,
        zorder=4,
    )
    ax.text(
        x + 0.16,
        y + h - 0.58,
        body,
        ha="left",
        va="top",
        fontsize=size,
        color=C_DARK,
        linespacing=1.12,
        zorder=4,
    )


def make_figure(
    out_dir: Path,
    site_static: Path | None = None,
    *,
    stem: str = "fig2_scCCVGBen_model_architecture",
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, AX_XMAX)
    ax.set_ylim(AX_YMIN, AX_YMAX)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _box(ax, 0.12, 0.24, 20.05, 7.20, "", fc=C_BG, ec="#E2E8F0", lw=1.0, radius=0.22)
    ax.text(0.28, 7.70, "scCCVGBen model architecture",
            fontsize=29.0, fontweight="bold", color=C_DARK, ha="left", va="center")
    ax.text(19.85, 7.70, "graph construction • encoder axis • dual reconstruction • BEN/DRE/LSE",
            fontsize=16.6, color=C_MUTED, ha="right", va="center")

    # Section envelopes make the architecture denser than a loose flowchart.
    sections = [
        (0.38, 0.66, 3.45, 6.64, C_BLUE, "A  input + graph"),
        (4.04, 0.66, 5.05, 6.64, C_ORANGE, "B  encoder plug-ins"),
        (9.32, 0.66, 6.40, 6.64, C_TEAL, "C  variational core"),
        (15.95, 0.66, 4.05, 6.64, C_PURPLE, "D  metric output"),
    ]
    for x, y, w, h, color, title in sections:
        _box(ax, x, y, w, h, "", fc="white", ec=color, lw=1.15, radius=0.18)
        ax.text(x + 0.22, y + h - 0.24, title, fontsize=18.0, color=color,
                fontweight="bold", ha="left", va="top")

    # A: data preparation and graph construction axis.
    _box(ax, 0.62, 6.05, 2.98, 0.58, "AnnData input\nX + optional labels", fc="#EFF6FF", ec=C_BLUE,
         color=C_DARK, size=14.4, weight="bold")
    _box(ax, 0.62, 5.22, 2.98, 0.58, "counts → log/HVG\nPCA feature matrix", fc="#FFFFFF", ec=C_BLUE,
         color=C_DARK, size=13.4)
    _box(ax, 0.62, 2.62, 2.98, 2.18, "", fc="#F0F9FF", ec=C_BLUE, lw=1.0)
    ax.text(0.84, 4.48, "5 graph choices", fontsize=15.6, fontweight="bold", color=C_BLUE)
    for i, method in enumerate(GRAPH_METHODS):
        ax.text(0.86, 4.08 - i * 0.33, f"• {method}", fontsize=13.0,
                color=C_DARK, ha="left", va="center")
    _box(ax, 0.58, 1.72, 3.06, 0.64, "cell graph A\n(edge_index / weight)",
         fc="#DBEAFE", ec=C_BLUE, color=C_BLUE, size=11.9, weight="bold")
    _box(ax, 0.68, 0.94, 2.86, 0.48, "same cells; graph axis", fc="#FFFFFF",
         ec="#93C5FD", color=C_MUTED, size=11.6)
    _arrow(ax, (2.11, 6.04), (2.11, 5.82), color=C_BLUE, lw=2.0)
    _arrow(ax, (2.11, 5.21), (2.11, 4.82), color=C_BLUE, lw=2.0)
    _arrow(ax, (1.00, 2.62), (1.00, 2.38), color=C_BLUE, lw=2.0)
    _arrow(ax, (3.62, 5.52), (4.02, 5.52), color=C_BLUE, lw=2.35,
           zorder=3.2, mutation_scale=18)
    _flow_label(ax, 3.94, 5.69, "X", color=C_BLUE)
    _arrow(ax, (3.66, 2.04), (4.02, 1.62), color=C_BLUE, lw=2.35, rad=-0.08,
           zorder=3.2, mutation_scale=18)
    _flow_label(ax, 3.96, 2.02, "A", color=C_BLUE)

    # B: encoder plug-in registry, tightly grouped by architectural family.
    ax.text(4.32, 6.42, "Encoder registry (14 plug-ins)", fontsize=17.2,
            color=C_ORANGE, fontweight="bold", ha="left", va="center")
    ax.text(4.32, 6.06, "fixed data/graph/losses; swap only fθ",
            fontsize=12.8, color=C_MUTED, ha="left", va="center")
    _pill_row(ax, ENCODERS[:4], 4.32, 5.18, 4.55, 0.76, cols=4, color=C_ORANGE, size=10.6)
    ax.text(4.34, 5.04, "attention family", fontsize=11.6, color=C_MUTED, ha="left")
    _pill_row(ax, ENCODERS[4:12], 4.32, 3.34, 4.55, 1.42, cols=4, color="#B45309", size=10.9)
    ax.text(4.34, 3.13, "message-passing / spectral family", fontsize=11.6, color=C_MUTED, ha="left")
    _pill_row(ax, ENCODERS[12:], 4.32, 2.45, 2.25, 0.62, cols=2, color=C_ORANGE, size=10.3)
    ax.text(6.82, 2.72, "+ edge/dynamic", fontsize=12.0,
            color=C_MUTED, ha="left", va="center")
    _box(ax, 4.30, 1.06, 4.60, 0.90,
         "plug-in contract: fθ(X, A) → qμ/qσ\nlosses + evaluation stay fixed",
         fc="#FFF7ED", ec="#FDBA74", color=C_DARK, size=12.0, ha="left")
    _arrow(ax, (8.84, 5.55), (9.60, 5.86), color=C_ORANGE, lw=2.35, rad=0.02,
           zorder=3.2, mutation_scale=18)
    _flow_label(ax, 9.15, 5.58, "fθ", color=C_ORANGE)

    # C: variational core and explicit dual reconstruction paths.
    ax.text(9.58, 6.42, "Variational graph autoencoder core", fontsize=17.0,
            color=C_TEAL, fontweight="bold", ha="left", va="center")
    _box(ax, 9.62, 5.55, 1.95, 0.62, "fθ(X, A)\ngraph encoder", fc="#ECFDF5", ec=C_TEAL,
         color=C_TEAL, size=13.4, weight="bold")
    _box(ax, 11.92, 5.55, 1.42, 0.62, "qμ, qσ", fc="#ECFDF5", ec=C_TEAL,
         color=C_TEAL, size=14.8, weight="bold")
    _box(ax, 13.58, 5.55, 0.76, 0.62, "z", fc="#D1FAE5", ec=C_TEAL,
         color=C_TEAL, size=18.4, weight="bold")
    _box(ax, 14.52, 5.55, 1.02, 0.62, "pred_x", fc="#EFF6FF", ec=C_BLUE,
         color=C_BLUE, size=13.8, weight="bold")
    _box(ax, 9.78, 4.12, 1.85, 0.64, "pred_a\nadj decoder", fc="#F0FDFA", ec=C_TEAL,
         color=C_TEAL, size=13.2, weight="bold")
    _box(ax, 12.00, 4.12, 1.35, 0.64, "KL", fc="#F0FDFA", ec=C_TEAL,
         color=C_TEAL, size=14.8, weight="bold")
    _box(ax, 13.58, 4.12, 0.76, 0.64, "i", fc="#FEF3C7", ec="#B45309",
         color="#92400E", size=18.4, weight="bold")
    _box(ax, 13.58, 2.80, 0.76, 0.64, "z′", fc="#D1FAE5", ec=C_TEAL,
         color=C_TEAL, size=18.4, weight="bold")
    _box(ax, 14.52, 2.80, 1.02, 0.64, "pred_xl", fc="#FFF7ED", ec=C_ORANGE,
         color=C_ORANGE, size=13.8, weight="bold")
    _box(ax, 9.76, 1.20, 5.95, 0.86,
         "loss = recon(pred_x, X) + inner-recon(pred_xl, X)\n+ adj(pred_a, A) + KL",
         fc="#FFFFFF", ec=C_LINE, color=C_DARK, size=12.9, ha="left")
    _box(ax, 9.68, 2.42, 3.02, 0.38, "adjacency path: A → pred_a", fc="#FFFFFF",
         ec="#A7F3D0", color=C_TEAL, size=10.9)
    _box(ax, 12.98, 2.42, 2.56, 0.38, "feature path: z / z′ → x", fc="#FFFFFF",
         ec="#FED7AA", color="#92400E", size=10.9)
    ax.text(12.88, 4.96, "latent sampling", fontsize=12.0, color=C_MUTED, ha="center")
    ax.text(
        14.05,
        3.68,
        "inner bottleneck",
        fontsize=10.9,
        color="#92400E",
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.10,rounding_size=0.04",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=4.4,
    )
    # Make the tensor logic explicit without letting connector strokes cover labels.
    _arrow(ax, (11.58, 5.86), (11.90, 5.86), color=C_TEAL, lw=2.2,
           zorder=3.1, mutation_scale=18)
    _arrow(ax, (13.35, 5.86), (13.56, 5.86), color=C_TEAL, lw=2.2,
           zorder=3.1, mutation_scale=18)
    _arrow(ax, (14.36, 5.86), (14.50, 5.86), color=C_BLUE, lw=2.2,
           zorder=3.1, mutation_scale=18)
    _arrow(ax, (13.96, 5.54), (13.96, 4.78), color="#B45309", lw=2.0,
           zorder=3.05, mutation_scale=17)
    _arrow(ax, (13.96, 4.10), (13.96, 3.46), color=C_TEAL, lw=2.0,
           zorder=3.05, mutation_scale=17)
    _arrow(ax, (14.36, 3.12), (14.50, 3.12), color=C_ORANGE, lw=2.2,
           zorder=3.1, mutation_scale=18)
    _path_arrow(ax, [(10.58, 5.52), (10.30, 5.18), (10.30, 4.78)],
                color=C_TEAL, lw=2.0, zorder=3.0, mutation_scale=17)
    _path_arrow(ax, [(10.70, 4.10), (10.02, 3.68), (10.02, 2.10)],
                color=C_TEAL, lw=2.0, zorder=2.65, mutation_scale=17)
    _path_arrow(ax, [(12.68, 4.10), (12.18, 3.68), (12.18, 2.10)],
                color=C_TEAL, lw=2.0, zorder=2.65, mutation_scale=17)
    _path_arrow(ax, [(15.02, 5.54), (15.62, 5.54), (15.62, 2.08)],
                color=C_BLUE, lw=2.0, zorder=2.65, mutation_scale=17)
    _path_arrow(ax, [(15.02, 2.78), (15.40, 2.78), (15.40, 2.08)],
                color=C_ORANGE, lw=2.0, zorder=2.65, mutation_scale=17)
    _path_arrow(ax, [(15.56, 5.86), (15.86, 5.86), (16.10, 5.62)],
                color=C_PURPLE, lw=2.25, zorder=2.55, mutation_scale=17)
    _path_arrow(ax, [(15.56, 3.12), (15.84, 3.12), (16.10, 4.22)],
                color=C_PURPLE, lw=2.25, zorder=2.55, mutation_scale=17)
    _path_arrow(ax, [(15.56, 1.64), (15.84, 1.64), (16.10, 2.18)],
                color=C_PURPLE, lw=2.25, zorder=2.55, mutation_scale=17)

    # D: output table; compact groups retain the 20-score structure without
    # adding a second title that competes with the section label.
    _box(ax, 16.18, 6.18, 3.55, 0.62, "metric table: 20 display scores\n3 BEN + 10 DRE + 7 LSE", fc="#F5F3FF",
         ec=C_PURPLE, color=C_PURPLE, size=13.0, weight="bold")
    _metric_family_card(
        ax, 16.18, 5.24, 3.55, 0.74,
        "BEN", "clustering compactness (3)",
        ", ".join(BEN_METRICS),
        size=10.6,
    )
    dre_text = (
        f"UMAP: {', '.join(DRE_METRICS)}\n"
        f"t-SNE: {', '.join(DRE_METRICS)}"
    )
    _metric_family_card(
        ax, 16.18, 3.56, 3.55, 1.34,
        "DRE", "embedding / coranking (10)",
        dre_text,
        size=10.25,
    )
    lse_text = f"{', '.join(LSE_METRICS)}"
    _metric_family_card(
        ax, 16.18, 1.56, 3.55, 1.68,
        "LSE", "intrinsic geometry (7)",
        lse_text,
        size=10.35,
    )
    _box(ax, 16.18, 0.92, 3.55, 0.38, "publication grid: 4 rows × 5 panels", fc="#FFFFFF",
         ec="#DDD6FE", color=C_MUTED, size=11.2)

    legend_items = [
        (0.62, C_BLUE, "graph axis"), (2.36, C_ORANGE, "encoder axis"),
        (4.18, C_TEAL, "model tensors"), (6.16, C_PURPLE, "metric output"),
    ]
    for x, c, txt in legend_items:
        _box(ax, x, 0.32, 0.25, 0.18, "", fc=c, ec=c, radius=0.04)
        ax.text(x + 0.34, 0.41, txt, fontsize=12.0, color=C_MUTED, ha="left", va="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    outputs = [png, pdf]

    if site_static is not None:
        site_static.mkdir(parents=True, exist_ok=True)
        site_png = site_static / png.name
        shutil.copy2(png, site_png)
        outputs.append(site_png)

    return outputs

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--out-dir", type=Path, default=root / "figures")
    parser.add_argument("--site-static", type=Path, default=root / "site" / "static" / "images")
    parser.add_argument("--no-site-copy", action="store_true")
    parser.add_argument("--stem", default="fig2_scCCVGBen_model_architecture")
    parser.add_argument(
        "--partial-ok",
        action="store_true",
        help="Accepted for orchestrator parity; this static diagram is all-or-fail.",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=0,
        help="Accepted for orchestrator parity; not used by this static diagram.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    args = _parse_args(argv)
    site_static = None if args.no_site_copy else args.site_static
    outputs = make_figure(args.out_dir, site_static, stem=args.stem)
    for path in outputs:
        log.info("Saved %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
