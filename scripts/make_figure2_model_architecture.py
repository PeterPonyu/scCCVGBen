#!/usr/bin/env python3
"""Generate the architecture-only intermediate for manuscript Figure 2.

The canonical manuscript Figure 2 is the composite written by
``scripts/make_figure2_composite.py`` as
``figures/fig02_benchmark_architecture.{pdf,png}``. This module renders only
the panel-D architecture image (default stem ``fig02_architecture``) used by
that composite:
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
# Architecture diagram uses a single accent (slate) for borders and arrows, with
# black text. The original five-colour palette mapped each section to its own
# accent, but the resulting figure read as a colour catalogue rather than a
# diagram. The five aliases below now collapse onto two values so callers that
# pass `color=C_BLUE` etc. continue to work without code churn elsewhere.
C_DARK = "#1F2D3D"
C_MUTED = "#64748B"
C_ARROW = "#334155"
C_BG = "#F8FAFC"
C_CARD = "#FFFFFF"
C_LINE = "#CBD5E1"
C_ORANGE = C_DARK
C_BLUE = C_DARK
C_TEAL = C_DARK
C_PURPLE = C_DARK
C_GREEN = C_DARK

ENCODERS = [
    "GAT", "GATv2", "Trans-\nformer", "Super\nGAT",
    "GCN", "SAGE", "Graph", "Cheb", "TAG", "ARMA", "SG", "SSG",
    "GIN", "Edge\nConv",
]
GRAPH_METHODS = ["kNN-Euclid", "kNN-cosine", "SNN", "Mutual kNN", "Gaussian threshold"]
BEN_METRICS = ["ASW", "DAV", "CAL"]
DRE_METRICS = ["DC", "QL", "QG", r"$K_{\max}$", "Overall"]
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
         weight: str = "normal", ha: str = "center",
         zorder: float = 2) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
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


# Arrows are clamped below the small content boxes (zorder=2) so the rounded
# white-fill of every box hides any connector that would otherwise visually
# cross the text inside it. Chrome / section envelopes are at zorder=1.0 so
# arrows still draw on top of those backgrounds.
_ARROW_Z_CAP = 1.9


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
           *, color: str = C_ARROW, lw: float = 1.8, rad: float = 0.0,
           zorder: float = 1.85, mutation_scale: float = 15.0) -> None:
    # Section borders carry the semantic colours; connectors stay neutral so
    # the architecture reads as one continuous data-flow diagram.
    _ = color
    arr = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=C_ARROW,
        zorder=min(zorder, _ARROW_Z_CAP),
    )
    ax.add_patch(arr)


def _path_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str = C_ARROW,
    lw: float = 1.8,
    zorder: float = 1.85,
    mutation_scale: float = 15.0,
) -> None:
    """Draw a routed connector so arrows use whitespace corridors instead of labels."""
    _ = color
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    arr = FancyArrowPatch(
        path=MplPath(points, codes),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=C_ARROW,
        zorder=min(zorder, _ARROW_Z_CAP),
        capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(arr)


def _flow_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    size: float = 15.8,
) -> None:
    """Readable in-flow label for routed arrows."""
    _ = color
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=C_DARK,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.94,
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
        color=C_DARK,
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
    # Body sits right below the family/title header — keeps every word inside
    # its own card with a clean empty band underneath. Looser linespacing
    # avoids the visually-stacked feel where 2 body lines hug each other.
    ax.text(
        x + 0.16,
        y + h - 0.46,
        body,
        ha="left",
        va="top",
        fontsize=size,
        color=C_DARK,
        linespacing=1.40,
        zorder=4,
    )


def make_figure(
    out_dir: Path,
    site_static: Path | None = None,
    *,
    stem: str = "fig02_architecture",
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, AX_XMAX)
    ax.set_ylim(AX_YMIN, AX_YMAX)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _box(ax, 0.12, 0.24, 20.05, 7.20, "", fc=C_BG, ec="#E2E8F0", lw=1.0,
         radius=0.22, zorder=1.0)
    # Outer panel label "D" — architecture is the 4th panel in fig02 (A/B/C
    # are the metadata strip above). Section titles inside drop their letter
    # prefixes so there is no clash with the outer label.
    ax.text(0.06, 7.84, "D",
            fontsize=18.0, fontweight="bold", color=C_DARK, ha="left", va="top")
    # Section title "scCCVGBen model architecture" moved to the LaTeX figure
    # caption per paper-layout policy.
    # Long axis-by-axis description moved to the LaTeX figure caption.

    # Section envelopes make the architecture denser than a loose flowchart.
    sections = [
        (0.38, 0.66, 3.45, 6.64, C_BLUE, "Input + graph"),
        (4.04, 0.66, 5.05, 6.64, C_ORANGE, "Encoder plug-ins"),
        (9.32, 0.66, 6.40, 6.64, C_TEAL, "Variational core"),
        (15.95, 0.66, 4.05, 6.64, C_PURPLE, "Metric output"),
    ]
    for x, y, w, h, color, title in sections:
        _box(ax, x, y, w, h, "", fc="white", ec=color, lw=1.15, radius=0.18,
             zorder=1.0)
        ax.text(x + 0.22, y + h - 0.24, title, fontsize=17.0, color=C_DARK,
                fontweight="bold", ha="left", va="top")

    # A: data preparation and graph construction axis.
    _box(ax, 0.62, 6.05, 2.98, 0.58, "AnnData input\nX + optional labels", fc=C_BG, ec=C_BLUE,
         color=C_DARK, size=14.4, weight="bold")
    _box(ax, 0.62, 5.22, 2.98, 0.58, r"counts $\rightarrow$ log/HVG" + "\nPCA feature matrix", fc="#FFFFFF", ec=C_BLUE,
         color=C_DARK, size=13.4)
    _box(ax, 0.62, 2.62, 2.98, 2.18, "", fc=C_BG, ec=C_BLUE, lw=1.0)
    ax.text(0.84, 4.48, "5 graph choices", fontsize=14.0, fontweight="bold", color=C_DARK)
    for i, method in enumerate(GRAPH_METHODS):
        ax.text(0.86, 4.08 - i * 0.33, f"• {method}", fontsize=13.5,
                color=C_DARK, ha="left", va="center")
    _box(ax, 0.58, 1.72, 3.06, 0.64, "cell graph A\n(edge_index / weight)",
         fc="#E2E8F0", ec=C_BLUE, color=C_DARK, size=12.4, weight="bold")
    _box(ax, 0.68, 0.94, 2.86, 0.40, "same cells; graph axis", fc="#FFFFFF",
         ec=C_LINE, color=C_MUTED, size=11.6)
    _arrow(ax, (2.11, 6.04), (2.11, 5.82), color=C_BLUE, lw=2.0)
    _arrow(ax, (2.11, 5.21), (2.11, 4.82), color=C_BLUE, lw=2.0)
    _arrow(ax, (1.00, 2.62), (1.00, 2.38), color=C_BLUE, lw=2.0)
    _arrow(ax, (3.60, 5.52), (4.04, 5.52), color=C_BLUE, lw=1.9,
           zorder=3.2, mutation_scale=12)
    # Letter offset above the arrow path so the bbox-less label has clean air
    # around it instead of sitting on top of the connector stroke.
    _flow_label(ax, 3.82, 5.82, r"$X$", color=C_BLUE)
    _arrow(ax, (3.64, 2.04), (4.04, 1.62), color=C_BLUE, lw=1.9, rad=-0.08,
           zorder=3.2, mutation_scale=12)
    # The graph tensor is already named in the source/contract boxes; omitting
    # an extra in-flow "A" avoids crowding the input/encoder section boundary.

    # B: encoder plug-in registry, tightly grouped by architectural family.
    # Section title is carried by the orange envelope ('encoder plug-ins')
    # so the inner banner is dropped; only the contract caption remains.
    _box(ax, 4.30, 6.08, 4.60, 0.54,
         r"fixed data / graph / losses" + "\n" + r"swap only encoder $f_\theta$",
         fc=C_BG, ec=C_LINE, color=C_DARK, size=11.5,
         weight="bold", ha="left")
    _pill_row(ax, ENCODERS[:4], 4.32, 5.18, 4.55, 0.76, cols=4, color=C_ORANGE, size=10.6)
    ax.text(4.34, 4.97, "attention family", fontsize=12.2, color=C_DARK,
            fontweight="bold", ha="left")
    _pill_row(ax, ENCODERS[4:12], 4.32, 3.34, 4.55, 1.42, cols=4, color=C_DARK, size=10.9)
    ax.text(4.34, 3.18, "message-passing / spectral family", fontsize=12.2,
            color=C_DARK, fontweight="bold", ha="left")
    _pill_row(ax, ENCODERS[12:], 4.32, 2.20, 2.25, 0.62, cols=2, color=C_ORANGE, size=10.3)
    ax.text(4.34, 2.04, "+ edge / dynamic", fontsize=12.2,
            color=C_DARK, fontweight="bold", ha="left", va="top")
    _box(ax, 4.30, 1.06, 4.60, 0.84,
         r"plug-in contract: $f_\theta(X, A) \rightarrow q_\mu / q_\sigma$" + "\n" +
         "losses + evaluation stay fixed",
         fc=C_BG, ec=C_LINE, color=C_DARK, size=12.0, ha="left")
    _arrow(ax, (8.90, 5.55), (9.62, 5.86), color=C_ORANGE, lw=1.9, rad=0.02,
           zorder=3.2, mutation_scale=12)
    _flow_label(ax, 9.22, 6.10, r"$f_\theta$", color=C_ORANGE)

    # C: variational core and explicit dual reconstruction paths.
    # Section title carried by the teal envelope ('variational core'); the
    # inner banner used to repeat the same idea, so it is dropped.
    _box(ax, 9.62, 5.55, 1.78, 0.62, r"$f_\theta(X, A)$" + "\ngraph encoder", fc=C_BG, ec=C_TEAL,
         color=C_DARK, size=13.8, weight="bold")
    _box(ax, 11.95, 5.55, 1.30, 0.62, r"$q_\mu,\ q_\sigma$", fc=C_BG, ec=C_TEAL,
         color=C_DARK, size=15.4, weight="bold")
    _box(ax, 13.78, 5.55, 0.62, 0.62, r"$z$", fc="#E2E8F0", ec=C_TEAL,
         color=C_DARK, size=16.0, weight="bold")
    _box(ax, 14.95, 5.55, 0.70, 0.62, r"$\hat{x}$", fc=C_BG, ec=C_BLUE,
         color=C_DARK, size=16.0, weight="bold")
    _box(ax, 9.78, 4.12, 1.85, 0.64, r"$\hat{A}$" + "\nadj decoder", fc=C_BG, ec=C_TEAL,
         color=C_DARK, size=13.6, weight="bold")
    _box(ax, 12.00, 4.12, 1.35, 0.64, r"$\mathrm{KL}$", fc=C_BG, ec=C_TEAL,
         color=C_DARK, size=15.4, weight="bold")
    _box(ax, 13.71, 4.12, 0.76, 0.64, r"$i$", fc="#F1F5F9", ec=C_LINE,
         color=C_DARK, size=16.0, weight="bold")
    _box(ax, 13.78, 2.80, 0.62, 0.64, r"$z'$", fc="#E2E8F0", ec=C_TEAL,
         color=C_DARK, size=16.0, weight="bold")
    _box(ax, 14.95, 2.80, 0.70, 0.64, r"$\hat{x}_\ell$", fc=C_BG, ec=C_ORANGE,
         color=C_DARK, size=16.0, weight="bold")
    _box(ax, 9.76, 1.20, 5.95, 0.86,
         r"$\mathcal{L} = \mathrm{recon}(\hat{x}, X) + \mathrm{inner\!\text{-}recon}(\hat{x}_\ell, X)$" + "\n" +
         r"$+\ \mathrm{adj}(\hat{A}, A) + \mathrm{KL}$",
         fc="#FFFFFF", ec=C_LINE, color=C_DARK, size=12.9, ha="left")
    # Path-family caption row pulled down a touch so it no longer shares its
    # top edge with the z′ / pred_xl tensor row above (was reading as one
    # merged stripe).
    _box(ax, 9.68, 2.32, 3.02, 0.38, r"adjacency path: $A \rightarrow \hat{A}$", fc="#FFFFFF",
         ec=C_LINE, color=C_DARK, size=10.9)
    _box(ax, 12.98, 2.32, 2.56, 0.38, r"feature path: $z / z' \rightarrow x$", fc="#FFFFFF",
         ec=C_LINE, color=C_DARK, size=10.9)
    # 'latent sampling' / 'inner bottleneck' descriptive labels removed —
    # they previously floated between two tensor rows and read as not
    # belonging to any specific box.
    # Inner-chain box positions are spread so each connector has at least
    # ~0.5 axis-units of body between adjacent box edges, giving the arrow a
    # visible shaft instead of just a hovering head. mutation_scale stays
    # modest (13-14) so heads do not collide across the variational core.
    _arrow(ax, (11.40, 5.86), (11.95, 5.86), color=C_TEAL, lw=1.8,
           zorder=3.1, mutation_scale=14)
    _arrow(ax, (13.25, 5.86), (13.78, 5.86), color=C_TEAL, lw=1.8,
           zorder=3.1, mutation_scale=14)
    _arrow(ax, (14.40, 5.86), (14.95, 5.86), color=C_BLUE, lw=1.8,
           zorder=3.1, mutation_scale=14)
    _arrow(ax, (14.09, 5.55), (14.09, 4.78), color=C_DARK, lw=1.8,
           zorder=3.05, mutation_scale=14)
    _arrow(ax, (14.09, 4.10), (14.09, 3.46), color=C_TEAL, lw=1.8,
           zorder=3.05, mutation_scale=14)
    _arrow(ax, (14.40, 3.12), (14.95, 3.12), color=C_ORANGE, lw=1.8,
           zorder=3.1, mutation_scale=14)
    _path_arrow(ax, [(10.51, 5.52), (10.30, 5.18), (10.30, 4.78)],
                color=C_TEAL, lw=2.0, zorder=3.0, mutation_scale=15)
    _path_arrow(ax, [(10.70, 4.10), (10.02, 3.68), (10.02, 2.10)],
                color=C_TEAL, lw=2.0, zorder=2.65, mutation_scale=15)
    _path_arrow(ax, [(12.68, 4.10), (12.18, 3.68), (12.18, 2.10)],
                color=C_TEAL, lw=2.0, zorder=2.65, mutation_scale=15)
    _path_arrow(ax, [(15.30, 5.54), (15.85, 5.54), (15.85, 2.10)],
                color=C_BLUE, lw=2.0, zorder=2.65, mutation_scale=15)
    _path_arrow(ax, [(15.30, 2.78), (15.55, 2.78), (15.55, 2.10)],
                color=C_ORANGE, lw=2.0, zorder=2.65, mutation_scale=15)
    _path_arrow(ax, [(15.65, 5.86), (15.92, 5.86), (16.10, 5.62)],
                color=C_PURPLE, lw=2.25, zorder=2.55, mutation_scale=15)
    _path_arrow(ax, [(15.65, 3.12), (15.92, 3.12), (16.10, 4.22)],
                color=C_PURPLE, lw=2.25, zorder=2.55, mutation_scale=15)
    _path_arrow(ax, [(15.71, 1.64), (15.92, 1.64), (16.10, 2.18)],
                color=C_PURPLE, lw=2.25, zorder=2.55, mutation_scale=15)

    # D: output table; compact groups retain the 20-score structure without
    # adding a second title that competes with the section label.
    _box(ax, 16.18, 6.28, 3.55, 0.48,
         "20 scores: BEN 3 | DRE 10 | LSE 7",
         fc=C_BG, ec=C_PURPLE, color=C_DARK, size=11.8, weight="bold")
    _metric_family_card(
        ax, 16.18, 5.10, 3.55, 1.10,
        "BEN", "clustering compactness (3)",
        ", ".join(BEN_METRICS),
        size=10.6,
    )
    dre_text = (
        f"UMAP: {', '.join(DRE_METRICS)}\n"
        f"t-SNE: {', '.join(DRE_METRICS)}"
    )
    _metric_family_card(
        ax, 16.18, 3.05, 3.55, 1.75,
        "DRE", "embedding / coranking (10)",
        dre_text,
        size=10.25,
    )
    lse_text = f"{', '.join(LSE_METRICS)}"
    _metric_family_card(
        ax, 16.18, 1.40, 3.55, 1.05,
        "LSE", "intrinsic geometry (7)",
        lse_text,
        size=10.35,
    )

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
    parser.add_argument("--stem", default="fig02_architecture")
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
