"""Generate fig14_runtime.{pdf,png} from training logs (best-effort).

Runtime metrics (per_epoch_s, total_time_s, peak_gpu_mem_mb) are not
currently persisted to result CSVs. This script attempts to recover them
by regex-scanning workspace/logs/axis_*_*.log. If zero rows are recovered,
a structural stub is rendered with axes labelled and the annotation
"Runtime metrics pending — to be regenerated when training-side
instrumentation lands."
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scccvgben.figures import apply_publication_rcparams  # noqa: E402

log = logging.getLogger(__name__)

EPOCH_RE = re.compile(r"epoch\s*=?\s*(\d+).*?(\d+\.\d+)\s*s", re.IGNORECASE)
TOTAL_RE = re.compile(r"total[_ ]time[_ ]?s?\s*[:=]\s*(\d+\.\d+)", re.IGNORECASE)
GPU_RE = re.compile(r"peak[_ ]?gpu[_ ]?mem[_ ]?(?:mb)?\s*[:=]\s*(\d+\.?\d*)", re.IGNORECASE)


def _parse_logs(log_dir: Path) -> pd.DataFrame:
    if not log_dir.is_dir():
        return pd.DataFrame()
    rows: list[dict] = []
    for log_file in sorted(log_dir.glob("axis_*_*.log")):
        try:
            text = log_file.read_text(errors="ignore")
        except OSError:
            continue
        per_epoch = [float(m.group(2)) for m in EPOCH_RE.finditer(text)]
        total = [float(m.group(1)) for m in TOTAL_RE.finditer(text)]
        gpu = [float(m.group(1)) for m in GPU_RE.finditer(text)]
        if not (per_epoch or total or gpu):
            continue
        axis = "A" if "axis_A" in log_file.name else ("B" if "axis_B" in log_file.name else "?")
        rows.append({
            "log": log_file.name,
            "axis": axis,
            "per_epoch_s": float(np.mean(per_epoch)) if per_epoch else np.nan,
            "total_time_s": float(np.mean(total)) if total else np.nan,
            "peak_gpu_mem_mb": float(np.max(gpu)) if gpu else np.nan,
        })
    return pd.DataFrame(rows)


def _render_stub(out_dir: Path) -> tuple[Path, Path]:
    apply_publication_rcparams()
    row_configs = [
        ("scRNA", "vary features", "Number of HVGs"),
        ("scATAC", "vary features", "Number of HVPs"),
        ("scRNA", "vary subgraph", "Subgraph size"),
        ("scATAC", "vary subgraph", "Subgraph size"),
    ]
    metrics = ("Per-epoch time (s)", "Total time (s)", "Peak GPU memory (MB)")
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), dpi=300)
    plt.subplots_adjust(hspace=0.42, wspace=0.32)
    for row_i, (modality, experiment, xlabel) in enumerate(row_configs):
        for col_j, title in enumerate(metrics):
            ax = axes[row_i, col_j]
            ax.text(
                0.5,
                0.5,
                "Runtime metrics pending\nregenerate after instrumentation lands",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="#475569",
            )
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[["top", "right"]].set_visible(False)
        axes[row_i, 0].text(
            -0.18,
            1.12,
            chr(ord("A") + row_i),
            transform=axes[row_i, 0].transAxes,
            fontsize=26,
            fontweight="bold",
            va="top",
        )
        axes[row_i, 1].set_title(f"{modality}: {experiment}", fontsize=14, pad=10)
    fig.suptitle("Fig 14 — runtime profiling scaffold (PRELIMINARY: data unavailable)",
                 fontsize=15, y=0.995)
    pdf_path = out_dir / "fig14_runtime.PRELIMINARY.pdf"
    png_path = out_dir / "fig14_runtime.PRELIMINARY.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path

def _render_real(df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    apply_publication_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, title in zip(
        axes,
        ("per_epoch_s", "total_time_s", "peak_gpu_mem_mb"),
        ("Per-epoch time (s)", "Total time (s)", "Peak GPU memory (MB)"),
    ):
        sub = df.dropna(subset=[col])
        if sub.empty:
            ax.text(0.5, 0.5, f"{col}: not recovered",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#475569")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.bar(sub["axis"], sub[col], color="#1f5f9f", alpha=0.85)
            ax.set_ylabel(col)
        ax.set_title(title)
    fig.suptitle("Fig 14 — runtime profiling (PRELIMINARY: log-derived)",
                 fontsize=12, y=1.0)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    fig.tight_layout()
    pdf_path = out_dir / "fig14_runtime.PRELIMINARY.pdf"
    png_path = out_dir / "fig14_runtime.PRELIMINARY.png"
    fig.savefig(pdf_path); fig.savefig(png_path)
    plt.close(fig)
    return pdf_path, png_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path,
                        default=REPO_ROOT / "workspace" / "logs")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--partial-ok", action="store_true")
    parser.add_argument("--target-n", type=int, default=1,
                        help="Accepted for orchestrator parity.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_publication_rcparams()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _parse_logs(args.log_dir)
    if df.empty:
        pdf_path, png_path = _render_stub(args.out_dir)
        log.info("fig14: structural stub (no runtime data recovered)")
    else:
        pdf_path, png_path = _render_real(df, args.out_dir)
        log.info("fig14: rendered from %d log row(s)", len(df))
    log.info("wrote %s", pdf_path)
    log.info("wrote %s", png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
