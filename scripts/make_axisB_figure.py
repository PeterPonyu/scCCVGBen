"""Generate fig_axisB_graph_robustness.{pdf,png} from results/graph_sweep/*.csv.

Box + strip overlay across 5 graph-construction methods (kNN-Euc baseline from
encoder_sweep + 4 alternatives from graph_sweep), with Wilcoxon Holm-corrected
significance brackets vs the kNN-Euc reference.

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
    METRIC_FAMILY_TITLES,
    METRIC_LABELS,
    METRIC_PANEL_GRID,
    NUMERIC_METRICS,
    add_method_display,
    apply_publication_rcparams,
    available_numeric_metrics,
    create_metric_grid_figure,
    filter_to_manifest,
    metric_coverage_audit,
    preliminary_path,
    short_method_name,
    write_metric_audit,
)
from scccvgben.figures._long_form import melt_sweep  # noqa: E402

log = logging.getLogger(__name__)

PRIMARY_METRICS = NUMERIC_METRICS
GRAPH_ORDER = (
    "scCCVGBen_GAT_kNN_euc",
    "scCCVGBen_GAT_kNN_cosine",
    "scCCVGBen_GAT_snn",
    "scCCVGBen_GAT_mutual_knn",
    "scCCVGBen_GAT_gaussian_threshold",
)
KNN_EUC_BASELINE_SOURCE = "scCCVGBen_GAT"
REFERENCE_METHOD = "scCCVGBen_GAT_kNN_euc"
DEFAULT_TARGET = 100


def _inject_kNN_euc_baseline(graph_long: pd.DataFrame, encoder_dir: Path) -> pd.DataFrame:
    """Pull scCCVGBen_GAT rows from encoder_sweep, rebrand as scCCVGBen_GAT_kNN_euc."""
    if not encoder_dir.is_dir():
        return graph_long
    enc_long = melt_sweep(encoder_dir, modality="scrna",
                          metrics=PRIMARY_METRICS)
    if enc_long.empty:
        return graph_long
    base = enc_long[enc_long["method"] == KNN_EUC_BASELINE_SOURCE].copy()
    base["method"] = REFERENCE_METHOD
    return pd.concat([base, graph_long], ignore_index=True)


def _attach_modality(long_df: pd.DataFrame, manifest: Path) -> pd.DataFrame:
    if not manifest.exists():
        return long_df
    m = pd.read_csv(manifest, usecols=["filename_key", "modality"])
    lookup = dict(zip(m["filename_key"], m["modality"]))
    long_df = long_df.copy()
    long_df["modality"] = long_df["dataset_id"].map(lookup).fillna(long_df["modality"])
    return long_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path,
                        default=REPO_ROOT / "results" / "graph_sweep")
    parser.add_argument("--encoder-dir", type=Path,
                        default=REPO_ROOT / "results" / "encoder_sweep")
    parser.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "data" / "benchmark_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--partial-ok", action="store_true")
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

    long_df = _inject_kNN_euc_baseline(long_df, args.encoder_dir)
    long_df = filter_to_manifest(long_df, args.manifest, modality="scrna")
    long_df = _attach_modality(long_df, args.manifest)
    long_df = add_method_display(long_df)

    methods_present = [m for m in GRAPH_ORDER if m in long_df["method"].unique()]
    if not methods_present:
        log.error("no graph methods recognised in long_df")
        return 1
    long_df = long_df[long_df["method"].isin(methods_present)].copy()

    n_datasets = long_df["dataset_id"].nunique()
    log.info("graphs: %d, datasets: %d (target %d)",
             len(methods_present), n_datasets, args.target_n)

    if n_datasets < args.target_n and not args.partial_ok:
        log.error("only %d/%d datasets — pass --partial-ok",
                  n_datasets, args.target_n)
        return 1

    metrics = available_numeric_metrics(long_df, PRIMARY_METRICS)
    log.info("metric coverage: %d/%d display metrics", len(metrics), len(PRIMARY_METRICS))
    reference = short_method_name(args.reference) if args.reference in methods_present else None
    method_order = [short_method_name(m) for m in methods_present]
    audit = metric_coverage_audit(
        long_df,
        PRIMARY_METRICS,
        figure_id="axisB",
        group_col="method_display",
        expected_datasets=args.target_n,
        expected_methods=len(method_order),
    )
    write_metric_audit(args.out_dir / "_metric_audit.csv", audit, figure_id="axisB")

    fig, _ = create_metric_grid_figure(
        long_df,
        metric_grid=METRIC_PANEL_GRID,
        group_col="method_display",
        reference_method=reference,
        method_order=method_order,
        family_titles=METRIC_FAMILY_TITLES,
        title=None,
        subtitle=None,
        metric_labels=METRIC_LABELS,
        per_col_width=3.02,
        per_row_height=3.20,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_axisB_graph_robustness"
    pdf_path = args.out_dir / preliminary_path(stem, n_datasets, args.target_n,
                                               suffix=".pdf").name
    png_path = args.out_dir / preliminary_path(stem, n_datasets, args.target_n,
                                               suffix=".png").name
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    log.info("wrote %s", pdf_path)
    log.info("wrote %s", png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
