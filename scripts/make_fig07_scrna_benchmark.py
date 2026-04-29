"""Generate fig07_scrna_benchmark.{pdf,png} — scRNA modality slice of Axis C.

13-method DR benchmark on the 100 scRNA datasets, reconciled with reused
CG_dl_merged baselines. Box + strip + Wilcoxon Holm-corrected brackets vs the
scCCVGBen flagship encoder.
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
    palette_for_methods,
    melt_reconciled,
    preliminary_path,
    short_method_name,
    write_metric_audit,
)
from scccvgben.figures.metrics import LOWER_IS_BETTER  # noqa: E402

log = logging.getLogger(__name__)

PRIMARY_METRICS = NUMERIC_METRICS
REFERENCE_METHODS = ("scCCVGBen",)
METHOD_CANDIDATES = (
    ("scCCVGBen", ("scCCVGBen",)),
    ("PCA", ("PCA",)),
    ("KPCA", ("KPCA",)),
    ("ICA", ("ICA",)),
    ("FA", ("FA",)),
    ("NMF", ("NMF",)),
    ("TSVD", ("TSVD",)),
    ("DICL", ("DICL",)),
    ("scVI", ("scVI",)),
    ("DIP", ("DIP", "DIPVAE")),
    ("INFO", ("INFO", "InfoVAE")),
    ("TC", ("TC", "TCVAE")),
    ("highBeta", ("highBeta", "HighBetaVAE")),
)
DEFAULT_TARGET = 100  # scRNA only
# Show all significant reference-vs-baseline calls in Fig. 7. Dense panels use
# compact per-method stars instead of a tall bracket stack, so markers remain
# visible without covering the box/strip plots.
SIGNIFICANCE_PAIRS_PER_PANEL: int | None = None


def _select_raw_methods(long_df) -> list[str]:
    """Pick one raw source row per public display method."""
    present = set(long_df["method"].dropna().astype(str).unique())
    selected: list[str] = []
    for _display, candidates in METHOD_CANDIDATES:
        for raw in candidates:
            if raw in present:
                selected.append(raw)
                break
    return selected


def _ranked_display_order(long_df, raw_methods: list[str]) -> list[str]:
    """Order methods weak-to-strong by mean rank across directional metrics."""
    display_order = []
    for raw in raw_methods:
        display = short_method_name(raw)
        if display not in display_order:
            display_order.append(display)
    if not display_order:
        return []

    ranking_metrics = [
        metric for metric in PRIMARY_METRICS
        if not metric.startswith("K_max")
    ]
    rank_series = []
    for metric in ranking_metrics:
        sub = long_df.loc[long_df["metric"] == metric]
        values = sub.groupby("method_display", observed=True)["value"].mean()
        values = values.reindex(display_order).dropna()
        if values.empty:
            continue
        rank_series.append(
            values.rank(ascending=metric in LOWER_IS_BETTER, method="average")
            .rename(metric)
        )

    if not rank_series:
        ordered = display_order
    else:
        mean_rank = pd.concat(rank_series, axis=1).mean(axis=1)
        ordered = mean_rank.sort_values(ascending=False).index.tolist()
        missing = [method for method in display_order if method not in ordered]
        ordered = missing + ordered

    if "scCCVGBen" in ordered:
        ordered = [method for method in ordered if method != "scCCVGBen"]
        ordered.append("scCCVGBen")
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scrna-dir", type=Path,
                        default=REPO_ROOT / "results" / "reconciled" / "scrna")
    parser.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "data" / "benchmark_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--partial-ok", action="store_true")
    parser.add_argument("--reference", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_publication_rcparams()

    if not args.scrna_dir.is_dir():
        log.error("reconciled scrna dir missing: %s", args.scrna_dir)
        return 1

    long_df = melt_reconciled(args.scrna_dir, modality="scrna",
                              metrics=PRIMARY_METRICS)
    if long_df.empty:
        log.error("no rows from %s", args.scrna_dir)
        return 1
    long_df = filter_to_manifest(long_df, args.manifest, modality="scrna")

    methods_present = _select_raw_methods(long_df)
    if not methods_present:
        log.error("no recognised methods in long_df")
        return 1
    long_df = long_df[long_df["method"].isin(methods_present)].copy()
    long_df = add_method_display(long_df)

    n_datasets = long_df["dataset_id"].nunique()
    log.info("methods: %d, scrna datasets: %d (target %d)",
             len(methods_present), n_datasets, args.target_n)

    if n_datasets < args.target_n and not args.partial_ok:
        log.error("only %d/%d datasets — pass --partial-ok",
                  n_datasets, args.target_n)
        return 1

    metrics = available_numeric_metrics(long_df, PRIMARY_METRICS)
    log.info("metric coverage: %d/%d display metrics", len(metrics), len(PRIMARY_METRICS))
    reference_raw = args.reference if args.reference in methods_present else None
    if reference_raw is None:
        reference_raw = next((m for m in REFERENCE_METHODS if m in methods_present), None)
    reference = short_method_name(reference_raw) if reference_raw else None
    method_order = _ranked_display_order(long_df, methods_present)
    log.info("method order (weak→strong): %s", " < ".join(method_order))
    audit = metric_coverage_audit(
        long_df,
        PRIMARY_METRICS,
        figure_id="fig07",
        group_col="method_display",
        expected_datasets=args.target_n,
        expected_methods=len(method_order),
    )
    write_metric_audit(args.out_dir / "_metric_audit.csv", audit, figure_id="fig07")

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
        per_col_width=3.34,
        per_row_height=3.46,
        significance_pairs_per_panel=SIGNIFICANCE_PAIRS_PER_PANEL,
        significance_dense_marker_threshold=5,
        significance_marker_fontsize=8.8,
        significance_step_fraction=0.105,
        xtick_rotation=78,
        hspace=0.85,
        palette=palette_for_methods(method_order),
        box_width=0.72,
        strip_size=2.75,
        strip_jitter=0.24,
        strip_alpha=0.60,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig07_scrna_benchmark"
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
