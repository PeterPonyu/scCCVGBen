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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

log = logging.getLogger(__name__)

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
    "GAT", "GATv2", "Transformer", "SuperGAT",
    "GCN", "SAGE", "Graph", "Cheb", "TAG", "ARMA", "SG", "SSG",
    "GIN", "EdgeConv",
]
GRAPH_METHODS = ["kNN_euclidean", "kNN_cosine", "snn", "mutual_knn", "gaussian_threshold"]
CLUSTERING = ["ASW", "DAV", "CAL", "COR", "NMI", "ARI"]
DRE = ["UMAP: corr, Q_local, Q_global, Kmax, overall", "t-SNE: corr, Q_local, Q_global, Kmax, overall"]
INTRINSIC = [
    "dimensionality", "spectral decay", "participation ratio", "anisotropy",
    "trajectory", "noise resilience", "core quality", "overall", "data type", "interpretation",
]

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Liberation Sans",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_W = 20.2
FIG_H = 8.45
AX_XMAX = 20.35
AX_YMAX = 8.38


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
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        fontweight=weight,
        linespacing=1.2,
        zorder=3,
    )
    return patch


def _label(ax: plt.Axes, x: float, y: float, text: str, *, size: float = 10,
           color: str = C_DARK, weight: str = "bold", ha: str = "left") -> None:
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
            fontweight=weight, zorder=4)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
           *, color: str = C_MUTED, lw: float = 1.8, rad: float = 0.0) -> None:
    arr = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        zorder=5,
    )
    ax.add_patch(arr)


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


def make_figure(out_dir: Path, site_static: Path | None = None) -> list[Path]:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, AX_XMAX)
    ax.set_ylim(0, AX_YMAX)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Soft background ribbons.
    _box(ax, 0.15, 0.28, 19.7, 7.20, "", fc=C_BG, ec="#E2E8F0", lw=1.0, radius=0.22)
    ax.text(0.30, 7.88, "scCCVGBen model architecture",
            fontsize=21.5, fontweight="bold", color=C_DARK, ha="left", va="center")
    ax.text(19.70, 7.88, "graph construction • encoder axis • dual reconstruction • metrics",
            fontsize=12.0, color=C_MUTED, ha="right", va="center")

    # Column headings.
    _label(ax, 0.55, 7.05, "A  Input + graph construction", color=C_BLUE, size=13.2)
    _label(ax, 4.45, 7.05, "B  Encoder plugin axis", color=C_ORANGE, size=13.2)
    _label(ax, 10.00, 7.05, "C  Variational core + dual reconstruction", color=C_TEAL, size=13.0)
    _label(ax, 16.88, 7.05, "D  26-metric output", color=C_PURPLE, size=13.2)

    # A: data and graph construction.
    _box(ax, 0.55, 6.00, 3.05, 0.66, "AnnData input\nX + optional labels", fc="#EFF6FF", ec=C_BLUE,
         color=C_DARK, size=11.4, weight="bold")
    _box(ax, 0.55, 5.05, 3.05, 0.66, "Counts layer → log/HVG\nPCA feature matrix", fc="#FFFFFF", ec=C_BLUE,
         color=C_DARK, size=10.5)
    _arrow(ax, (2.08, 5.99), (2.08, 5.73), color=C_BLUE)
    _arrow(ax, (2.08, 5.04), (2.08, 4.72), color=C_BLUE)
    _box(ax, 0.48, 2.42, 3.20, 2.05, "", fc="#F0F9FF", ec=C_BLUE, lw=1.1)
    ax.text(0.67, 4.17, "5 graph choices", fontsize=11.8, fontweight="bold", color=C_BLUE)
    for i, method in enumerate(GRAPH_METHODS):
        yy = 3.82 - i * 0.30
        ax.text(0.72, yy, f"• {method}", fontsize=9.8, color=C_DARK, ha="left", va="center")
    _box(ax, 1.10, 1.58, 2.05, 0.58, "cell graph A", fc="#DBEAFE", ec=C_BLUE,
         color=C_BLUE, size=11.3, weight="bold")
    _arrow(ax, (2.08, 2.48), (2.08, 2.17), color=C_BLUE)

    # B: encoder axis.
    _box(ax, 4.25, 2.18, 4.95, 4.32, "", fc="#FFF7ED", ec=C_ORANGE, lw=1.2, radius=0.18)
    ax.text(4.50, 6.18, "Interchangeable encoder registry (14)", fontsize=12.6,
            color=C_ORANGE, fontweight="bold", ha="left", va="center")
    ax.text(4.50, 5.78, "same data/losses; swap only\nmessage-passing module",
            fontsize=9.9, color=C_MUTED, ha="left", va="center", linespacing=1.08)
    _pill_row(ax, ENCODERS[:4], 4.50, 5.00, 4.35, 0.64, cols=4, color=C_ORANGE, size=9.1)
    ax.text(4.50, 4.78, "attention family", fontsize=9.2, color=C_MUTED, ha="left")
    _pill_row(ax, ENCODERS[4:12], 4.50, 3.33, 4.35, 1.18, cols=4, color="#B45309", size=9.1)
    ax.text(4.50, 3.12, "message-passing / spectral family", fontsize=9.2, color=C_MUTED, ha="left")
    _pill_row(ax, ENCODERS[12:], 4.50, 2.53, 2.25, 0.50, cols=2, color=C_ORANGE, size=9.1)
    ax.text(7.04, 2.78, "+ edge/dynamic variants", fontsize=9.4, color=C_MUTED, ha="left", va="center")

    _arrow(ax, (3.15, 1.86), (4.22, 3.92), color=C_MUTED, rad=0.12)
    _arrow(ax, (3.55, 6.33), (4.22, 5.88), color=C_MUTED, rad=-0.08)

    # C: model core.
    _box(ax, 10.00, 5.75, 2.05, 0.64, "Graph encoder\nfθ(X, A)", fc="#ECFDF5", ec=C_TEAL,
         color=C_TEAL, size=10.8, weight="bold")
    _box(ax, 12.45, 5.75, 1.50, 0.64, "qμ, qσ", fc="#ECFDF5", ec=C_TEAL,
         color=C_TEAL, size=11.5, weight="bold")
    _box(ax, 14.23, 5.75, 0.92, 0.64, "z", fc="#D1FAE5", ec=C_TEAL,
         color=C_TEAL, size=14.0, weight="bold")
    _box(ax, 14.23, 4.37, 0.92, 0.64, "i", fc="#FEF3C7", ec="#B45309",
         color="#92400E", size=14.0, weight="bold")
    _box(ax, 14.23, 2.95, 0.92, 0.64, "z′", fc="#D1FAE5", ec=C_TEAL,
         color=C_TEAL, size=14.0, weight="bold")
    _box(ax, 10.00, 4.37, 2.02, 0.64, "A decoder\npred_a", fc="#F0FDFA", ec=C_TEAL,
         color=C_TEAL, size=10.6, weight="bold")
    _box(ax, 12.15, 4.37, 1.45, 0.64, "KL", fc="#F0FDFA", ec=C_TEAL,
         color=C_TEAL, size=11.2, weight="bold")
    _box(ax, 15.55, 5.75, 1.25, 0.64, "pred_x", fc="#EFF6FF", ec=C_BLUE,
         color=C_BLUE, size=11.2, weight="bold")
    _box(ax, 15.55, 2.95, 1.25, 0.64, "pred_xl", fc="#FFF7ED", ec=C_ORANGE,
         color=C_ORANGE, size=11.2, weight="bold")
    _box(ax, 9.75, 1.08, 7.10, 0.74,
         "loss = recon(pred_x, X) + inner-recon(pred_xl, X)\n+ KL + adj(pred_a, A)",
         fc="#FFFFFF", ec=C_LINE, color=C_DARK, size=9.8)

    _arrow(ax, (9.20, 4.18), (9.97, 6.04), color=C_TEAL, rad=0.08)
    _arrow(ax, (12.05, 6.07), (12.42, 6.07), color=C_TEAL)
    _arrow(ax, (13.95, 6.07), (14.20, 6.07), color=C_TEAL)
    _arrow(ax, (14.69, 5.74), (14.69, 5.03), color="#B45309")
    _arrow(ax, (14.69, 4.36), (14.69, 3.61), color=C_TEAL)
    _arrow(ax, (15.15, 6.07), (15.52, 6.07), color=C_BLUE)
    _arrow(ax, (15.15, 3.27), (15.52, 3.27), color=C_ORANGE)
    _arrow(ax, (10.55, 5.68), (10.95, 5.03), color=C_TEAL, rad=0.12)
    _arrow(ax, (11.02, 4.36), (11.02, 1.83), color=C_TEAL, rad=0.0)
    _arrow(ax, (16.18, 5.74), (15.70, 1.83), color=C_BLUE, rad=0.12)
    _arrow(ax, (16.18, 2.94), (15.70, 1.83), color=C_ORANGE, rad=-0.12)

    ax.text(12.20, 5.12, "latent sampling", fontsize=9.6, color=C_MUTED, ha="left")
    ax.text(15.12, 4.82, "inner bottleneck", fontsize=9.6, color="#92400E", ha="left")

    # D: output metrics.
    _box(ax, 17.15, 5.65, 2.45, 0.86, "metrics table\n26 scores", fc="#F5F3FF", ec=C_PURPLE,
         color=C_PURPLE, size=10.7, weight="bold")
    _box(ax, 17.00, 4.56, 2.85, 0.84, _wrapped("6 clustering: " + ", ".join(CLUSTERING), 34),
         fc="white", ec=C_PURPLE, color=C_DARK, size=8.9)
    dre_text = "10 DRE / coranking\nUMAP: corr, Q_local,\nQ_global, Kmax, overall\nt-SNE: corr, Q_local,\nQ_global, Kmax, overall"
    _box(ax, 17.00, 3.12, 2.85, 1.22, dre_text,
         fc="white", ec=C_PURPLE, color=C_DARK, size=8.3)
    intrinsic_text = "10 intrinsic geometry\n" + _wrapped(", ".join(INTRINSIC), 34)
    _box(ax, 17.00, 1.44, 2.85, 1.38, intrinsic_text,
         fc="white", ec=C_PURPLE, color=C_DARK, size=8.1)
    _arrow(ax, (16.82, 6.07), (17.13, 6.00), color=C_PURPLE)
    _arrow(ax, (16.82, 3.27), (17.00, 3.72), color=C_PURPLE, rad=0.12)

    # Bottom legend rails to distinguish benchmark axes from model tensors.
    legend_items = [
        (0.58, C_BLUE, "graph axis"), (2.10, C_ORANGE, "encoder axis"),
        (3.70, C_TEAL, "model tensors"), (5.35, C_PURPLE, "metric output"),
    ]
    for x, c, txt in legend_items:
          _box(ax, x, 0.58, 0.24, 0.18, "", fc=c, ec=c, radius=0.04)
          ax.text(x + 0.34, 0.67, txt, fontsize=9.7, color=C_MUTED, ha="left", va="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig2_scCCVGBen_model_architecture.png"
    pdf = out_dir / "fig2_scCCVGBen_model_architecture.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(pdf, dpi=300, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    outputs = [png, pdf]

    if site_static is not None:
        site_static.mkdir(parents=True, exist_ok=True)
        site_png = site_static / png.name
        shutil.copy2(png, site_png)
        outputs.append(site_png)

    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--out-dir", type=Path, default=root / "figures")
    parser.add_argument("--site-static", type=Path, default=root / "site" / "static" / "images")
    parser.add_argument("--no-site-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    args = _parse_args()
    site_static = None if args.no_site_copy else args.site_static
    outputs = make_figure(args.out_dir, site_static)
    for path in outputs:
        log.info("Saved %s", path)


if __name__ == "__main__":
    main()
