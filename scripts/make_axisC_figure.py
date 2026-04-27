"""Generate fig_axisC_baselines.{pdf,png} from results/reconciled/{scrna,scatac}/*.csv.

Axis C compares the scCCVGBen flagship encoder against 13 baseline DR methods
across both modalities. The reconciled CSVs already merge any trained Axis C
rows with reused CG_dl_merged baselines.

Box + strip overlay with Wilcoxon Holm-corrected brackets vs the scCCVGBen
flagship reference. The .PRELIMINARY. infix marks runs where < target
reconciled CSVs are present.
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
    melt_reconciled,
    preliminary_path,
    short_method_name,
    write_metric_audit,
)

log = logging.getLogger(__name__)

PRIMARY_METRICS = NUMERIC_METRICS
REFERENCE_METHODS = ("scCCVGBen_GAT",)
# Bare 'scCCVGBen' was the legacy CCVGAE-revised GAT row (epochs=200, i_dim=10);
# scCCVGBen_GAT is the current scccvgben GAT (epochs=100, i_dim=5). Plotting
# both was a hyperparameter-inconsistent duplicate. Stripped from reconciled
# data and dropped from the method list — only the canonical current row stays.
BASELINE_METHODS = (
    "scCCVGBen_GAT", "PCA", "KPCA", "ICA", "FA", "NMF", "TSVD",
    "DICL", "scVI", "DIP", "DIPVAE", "INFO", "InfoVAE", "TC", "TCVAE",
    "highBeta", "HighBetaVAE",
)
DEFAULT_TARGET = 200  # 100 scRNA + 100 scATAC


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scrna-dir", type=Path,
                        default=REPO_ROOT / "results" / "reconciled" / "scrna")
    parser.add_argument("--scatac-dir", type=Path,
                        default=REPO_ROOT / "results" / "reconciled" / "scatac")
    parser.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "data" / "benchmark_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--partial-ok", action="store_true")
    parser.add_argument("--reference", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_publication_rcparams()

    frames = []
    if args.scrna_dir.is_dir():
        frames.append(filter_to_manifest(
            melt_reconciled(args.scrna_dir, modality="scrna", metrics=PRIMARY_METRICS),
            args.manifest,
            modality="scrna",
        ))
    if args.scatac_dir.is_dir():
        frames.append(filter_to_manifest(
            melt_reconciled(args.scatac_dir, modality="scatac", metrics=PRIMARY_METRICS),
            args.manifest,
            modality="scatac",
        ))
    non_empty = [f for f in frames if not f.empty]
    long_df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    if long_df.empty:
        log.error("no rows from reconciled dirs")
        return 1

    methods_present = [m for m in BASELINE_METHODS if m in long_df["method"].unique()]
    if not methods_present:
        log.error("no recognised baseline methods in long_df")
        return 1
    long_df = long_df[long_df["method"].isin(methods_present)].copy()
    long_df = add_method_display(long_df)

    n_datasets = long_df["dataset_id"].nunique()
    log.info("methods: %d, datasets: %d (target %d)",
             len(methods_present), n_datasets, args.target_n)

    if n_datasets < args.target_n and not args.partial_ok:
        log.error("only %d/%d datasets — pass --partial-ok",
                  n_datasets, args.target_n)
        return 1

    metrics = available_numeric_metrics(long_df, PRIMARY_METRICS)
    log.info("metric coverage: %d/%d display metrics", len(metrics), len(PRIMARY_METRICS))
    requested_reference = args.reference
    reference_raw = requested_reference if requested_reference in methods_present else None
    if reference_raw is None:
        reference_raw = next((m for m in REFERENCE_METHODS if m in methods_present), None)
    reference = short_method_name(reference_raw) if reference_raw else None
    method_order = [short_method_name(m) for m in methods_present]
    audit = metric_coverage_audit(
        long_df,
        PRIMARY_METRICS,
        figure_id="axisC",
        group_col="method_display",
        expected_datasets=args.target_n,
        expected_methods=len(method_order),
    )
    write_metric_audit(args.out_dir / "_metric_audit.csv", audit, figure_id="axisC")

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
        per_col_width=3.22,
        per_row_height=3.38,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig_axisC_baselines"
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
