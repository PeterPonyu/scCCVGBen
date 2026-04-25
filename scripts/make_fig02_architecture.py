"""Generate fig02_architecture.{pdf,png} — scCCVGBen pipeline schematic.

Static box-and-arrow diagram showing the scCCVGBen reference pipeline plus the
two exploration arms this benchmark prepends:

  Input -> Preprocess -> [Axis A: 14 encoders] x [Axis B: 5 graphs]
        -> variational core (centroid + bottleneck) -> Decode -> Metrics

The figure carries no benchmark data, so the .PRELIMINARY. infix is never
used for fig02.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scccvgben.figures import apply_publication_rcparams  # noqa: E402

log = logging.getLogger(__name__)

PALETTE = {
    "input": "#94a3b8",
    "preprocess": "#cbd5e1",
    "axisA": "#7a5ab8",   # encoder arm (NEW in scCCVGBen)
    "axisB": "#24989f",   # graph arm (NEW in scCCVGBen)
    "core": "#1f5f9f",
    "metrics": "#b54848",
    "edge": "#475569",
}


def _box(ax: plt.Axes, x: float, y: float, w: float, h: float,
         label: str, color: str, *, fontsize: int = 10,
         boxstyle: str = "round,pad=0.06") -> None:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle=boxstyle,
            facecolor=color, edgecolor=PALETTE["edge"], linewidth=1.0,
            alpha=0.9,
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white")


def _arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "-|>",
            "color": PALETTE["edge"],
            "lw": 1.2,
            "shrinkA": 4, "shrinkB": 4,
        },
    )


def _draw_axisA_arm(ax: plt.Axes, x0: float, y0: float) -> None:
    encoders = ["GAT", "GATv2", "Trans.", "SuperGAT",
                "GCN", "SAGE", "GIN", "Cheb",
                "EdgeConv", "ARMA", "SG", "TAG", "Graph", "SSG"]
    cell_w, cell_h = 0.95, 0.6
    cols = 7
    for idx, enc in enumerate(encoders):
        col, row = idx % cols, idx // cols
        x = x0 + col * (cell_w + 0.08)
        y = y0 - row * (cell_h + 0.18)
        _box(ax, x, y, cell_w, cell_h, enc, PALETTE["axisA"], fontsize=8,
             boxstyle="round,pad=0.04")
    ax.text(x0 - 0.2, y0 + cell_h + 0.15,
            "Axis A — graph encoder zoo",
            fontsize=10, fontweight="bold", color=PALETTE["axisA"], ha="left")


def _draw_axisB_arm(ax: plt.Axes, x0: float, y0: float) -> None:
    graphs = ["kNN-Euc", "kNN-cos", "SNN", "Mutual-kNN", "Gaussian"]
    cell_w, cell_h = 1.5, 0.6
    for idx, g in enumerate(graphs):
        x = x0 + idx * (cell_w + 0.12)
        _box(ax, x, y0, cell_w, cell_h, g, PALETTE["axisB"], fontsize=9,
             boxstyle="round,pad=0.04")
    ax.text(x0 - 0.2, y0 + cell_h + 0.15,
            "Axis B — graph construction",
            fontsize=10, fontweight="bold", color=PALETTE["axisB"], ha="left")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--partial-ok", action="store_true",
                        help="Accepted for orchestrator parity; fig02 is data-free.")
    parser.add_argument("--target-n", type=int, default=0,
                        help="Accepted for orchestrator parity; not used.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_publication_rcparams()

    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    _box(ax, 0.4, 5.9, 1.8, 0.8, "Input\nh5ad", PALETTE["input"])
    _box(ax, 0.4, 4.5, 1.8, 0.8, "Preprocess\n(HVG / TF-IDF)", PALETTE["preprocess"])
    _arrow(ax, 1.3, 5.6, 1.3, 5.0)

    _draw_axisA_arm(ax, x0=2.7, y0=5.9)
    _draw_axisB_arm(ax, x0=2.7, y0=3.6)

    _arrow(ax, 2.2, 5.0, 2.7, 5.0)
    _arrow(ax, 2.2, 4.6, 2.7, 3.7)

    _box(ax, 10.0, 4.55, 2.5, 1.35,
         "Variational core\ncentroid + bottleneck", PALETTE["core"], fontsize=10)
    _arrow(ax, 9.5, 5.9, 10.0, 5.25)
    _arrow(ax, 9.5, 3.9, 10.0, 4.9)

    _box(ax, 10.0, 2.85, 2.5, 0.9,
         "Dual decode\nX and A", "#24989f", fontsize=10)
    _arrow(ax, 11.25, 4.55, 11.25, 3.75)

    _box(ax, 13.2, 4.7, 2.1, 0.8, "6 clustering\nmetrics", PALETTE["metrics"], fontsize=9)
    _box(ax, 13.2, 3.55, 2.1, 0.8, "10 DRE\nmetrics", PALETTE["metrics"], fontsize=9)
    _box(ax, 13.2, 2.40, 2.1, 0.8, "10 intrinsic\nfields", PALETTE["metrics"], fontsize=9)
    _arrow(ax, 12.5, 3.30, 13.2, 4.95)
    _arrow(ax, 12.5, 3.30, 13.2, 3.95)
    _arrow(ax, 12.5, 3.30, 13.2, 2.80)

    fig.suptitle("Fig 02 — scCCVGBen pipeline architecture", fontsize=14, y=0.98)
    fig.subplots_adjust(top=0.92)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / "fig02_architecture.pdf"
    png_path = args.out_dir / "fig02_architecture.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    log.info("wrote %s", pdf_path)
    log.info("wrote %s", png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
