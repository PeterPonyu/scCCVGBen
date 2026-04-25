"""Generate fig_axisA_encoder_ranking.{pdf,png} from results/encoder_sweep/*.csv.

Box + strip overlay across 14 graph encoders, with Wilcoxon Holm-corrected
significance brackets vs the scCCVGBen_GAT reference. Attention-family
encoders are placed left of message-passing encoders for visual grouping.

Output filename uses the .PRELIMINARY. infix when n_datasets < target.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scccvgben.figures import (  # noqa: E402
    METRIC_LABELS,
    NUMERIC_METRICS,
    add_method_display,
    apply_publication_rcparams,
    available_numeric_metrics,
    create_publication_figure,
    filter_to_manifest,
    preliminary_path,
    short_method_name,
)
from scccvgben.figures._long_form import melt_sweep  # noqa: E402

log = logging.getLogger(__name__)

PRIMARY_METRICS = NUMERIC_METRICS
ATTENTION_FAMILY = ("GAT", "GATv2", "Transformer", "SuperGAT")
REFERENCE_METHOD = "scCCVGBen_GAT"
DEFAULT_TARGET = 100


def _normalise_encoder(name: str) -> str:
    return name.replace("scCCVGBen_", "")


def _attach_modality(long_df: pd.DataFrame, manifest: Path) -> pd.DataFrame:
    if not manifest.exists():
        return long_df
    m = pd.read_csv(manifest, usecols=["filename_key", "modality"])
    lookup = dict(zip(m["filename_key"], m["modality"]))
    long_df = long_df.copy()
    long_df["modality"] = long_df["dataset_id"].map(lookup).fillna(long_df["modality"])
    return long_df


def _method_order(present_methods: list[str]) -> list[str]:
    short = {m: _normalise_encoder(m) for m in present_methods}
    attention = [m for m, s in short.items() if s in ATTENTION_FAMILY]
    other = [m for m in present_methods if m not in attention]
    attention.sort(key=lambda m: ATTENTION_FAMILY.index(short[m]))
    other.sort()
    return attention + other


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path,
                        default=REPO_ROOT / "results" / "encoder_sweep")
    parser.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "data" / "benchmark_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--partial-ok", action="store_true",
                        help="Render even when fewer than target-n datasets exist.")
    parser.add_argument("--reference", default=REFERENCE_METHOD)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_publication_rcparams()

    if not args.sweep_dir.is_dir():
        log.error("sweep dir missing: %s", args.sweep_dir)
        return 1

    long_df = melt_sweep(args.sweep_dir, modality="scrna",
                         metrics=PRIMARY_METRICS)
    if long_df.empty:
        log.error("no rows from %s", args.sweep_dir)
        return 1
    long_df = filter_to_manifest(long_df, args.manifest, modality="scrna")
    long_df = _attach_modality(long_df, args.manifest)
    long_df = add_method_display(long_df)

    methods = sorted(long_df["method"].dropna().unique().tolist())
    if not methods:
        log.error("no methods in long_df")
        return 1
    methods = _method_order(methods)
    n_datasets = long_df["dataset_id"].nunique()

    log.info("encoders: %d, datasets: %d (target %d)",
             len(methods), n_datasets, args.target_n)

    if n_datasets < args.target_n and not args.partial_ok:
        log.error("only %d/%d datasets — pass --partial-ok to render anyway",
                  n_datasets, args.target_n)
        return 1

    metrics = available_numeric_metrics(long_df, PRIMARY_METRICS)
    reference = short_method_name(args.reference) if args.reference in methods else None
    method_order = [short_method_name(m) for m in methods]
    fig, _ = create_publication_figure(
        long_df,
        metrics=metrics,
        group_col="method_display",
        reference_method=reference,
        method_order=method_order,
        ncols=4,
        figsize=(18, 24),
        strip_size=1.4,
        strip_alpha=0.42,
        metric_labels=METRIC_LABELS,
    )
    if n_datasets < args.target_n:
        fig.suptitle(
            f"Axis A — encoder ranking across {len(metrics)} numeric metrics "
            f"(PRELIMINARY: {n_datasets}/{args.target_n} datasets)",
            fontsize=14, y=1.0,
        )
    else:
        fig.suptitle(
            f"Axis A — encoder ranking across {len(metrics)} numeric metrics",
            fontsize=14, y=1.0,
        )
    fig.subplots_adjust(top=0.965)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_axisA_encoder_ranking"
    pdf_name = preliminary_path(stem, n_datasets, args.target_n, suffix=".pdf").name
    png_name = preliminary_path(stem, n_datasets, args.target_n, suffix=".png").name
    pdf_path = args.out_dir / pdf_name
    png_path = args.out_dir / png_name

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    log.info("wrote %s", pdf_path)
    log.info("wrote %s", png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
