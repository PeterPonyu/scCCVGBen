"""Generate fig10-12 case-study metric panels.

The case-study datasets are placeholders until the biological narratives are
pinned, but the plotting contract is final: every available numeric metric is
shown and each method is centered against the scCCVGBen reference row where the
reference exists. Positive heatmap values therefore mean better-than-reference
for higher-is-better metrics and lower-than-reference for DAV.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scccvgben.data.result_csv_normalizer import load_reused_csv  # noqa: E402
from scccvgben.figures import apply_publication_rcparams, dataset_key_from_result_stem  # noqa: E402
from scccvgben.figures.metrics import (  # noqa: E402
    LOWER_IS_BETTER,
    METRIC_FAMILY_ROWS,
    METRIC_LABELS,
    NUMERIC_METRICS,
    short_method_name,
)

log = logging.getLogger(__name__)

REFERENCE_CANDIDATES = ("scCCVGBen", "scCCVGBen_GAT")


def _available_result_index(scrna_dir: Path, scatac_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    manifest = manifest.copy()
    manifest["cell_count"] = pd.to_numeric(manifest["cell_count"], errors="coerce")
    meta = manifest.set_index("filename_key", drop=False)
    for modality, directory in (("scrna", scrna_dir), ("scatac", scatac_dir)):
        if not directory.is_dir():
            continue
        for csv_path in sorted(directory.glob("*.csv")):
            key = dataset_key_from_result_stem(csv_path.stem)
            if key not in meta.index:
                continue
            item = meta.loc[key].to_dict()
            rows.append({
                "dataset_id": key,
                "modality": modality,
                "csv_path": csv_path,
                "cell_count": item.get("cell_count"),
                "tissue": item.get("tissue", "unknown"),
            })
    return pd.DataFrame(rows)


def _pick_cases(index: pd.DataFrame) -> pd.DataFrame:
    if index.empty:
        return index
    index = index.dropna(subset=["cell_count"]).sort_values("cell_count")
    if len(index) <= 3:
        return index
    picks = [index.iloc[0], index.iloc[len(index) // 2], index.iloc[-1]]
    return pd.DataFrame(picks).drop_duplicates(subset=["dataset_id"])


def _metric_delta(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    df = df.copy()
    method_col = "method" if "method" in df.columns else df.columns[0]
    df = df.rename(columns={method_col: "method"})
    present = [m for m in metrics if m in df.columns]
    wide = df[["method", *present]].copy()
    for metric in present:
        wide[metric] = pd.to_numeric(wide[metric], errors="coerce")
    reference = next((m for m in REFERENCE_CANDIDATES if m in set(wide["method"])), None)
    if reference is None:
        values = wide.set_index("method")[present]
        return (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    ref = wide.loc[wide["method"] == reference, present].iloc[0]
    delta = wide.set_index("method")[present].subtract(ref, axis=1)
    for metric in present:
        if metric in LOWER_IS_BETTER:
            delta[metric] = -delta[metric]
    return delta


def _render_case(row: pd.Series, fig_id: str, out_dir: Path) -> Path:
    apply_publication_rcparams()
    raw = load_reused_csv(Path(row["csv_path"]), modality=str(row["modality"]))
    metrics = [m for m in NUMERIC_METRICS if m in raw.columns]
    delta = _metric_delta(raw, metrics).dropna(axis=1, how="all").dropna(axis=0, how="all")
    if delta.empty:
        raise ValueError(f"no numeric metrics for {row['csv_path']}")
    display_index = [short_method_name(str(m)) for m in delta.index]
    labels = [METRIC_LABELS.get(m, m) for m in delta.columns]

    vmax = float(np.nanpercentile(np.abs(delta.to_numpy(dtype=float)), 95))
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(17.5, 6.2), dpi=300)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#F8FAFC")
    values = np.ma.masked_invalid(delta.to_numpy(dtype=float))
    im = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(display_index)))
    ax.set_yticklabels(display_index, fontsize=10)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(display_index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    metric_to_family = {
        metric: family for family, family_metrics in METRIC_FAMILY_ROWS for metric in family_metrics
    }
    family_runs: list[tuple[str, int, int]] = []
    start = 0
    current = metric_to_family.get(str(delta.columns[0]), "metrics")
    for idx, metric in enumerate(delta.columns[1:], start=1):
        family = metric_to_family.get(str(metric), "metrics")
        if family == current:
            continue
        family_runs.append((current, start, idx - 1))
        ax.axvline(idx - 0.5, color="#334155", linewidth=0.9, alpha=0.55)
        start = idx
        current = family
    family_runs.append((current, start, len(delta.columns) - 1))
    for family, first, last in family_runs:
        ax.text(
            (first + last) / 2,
            1.035,
            family,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#334155",
        )

    ax.set_title(
        f"{fig_id} — {row['modality']} case-study deltas across BEN / DRE / LSE metrics\n"
        f"{row['dataset_id']} · tissue={row['tissue']} · cells={int(row['cell_count']):,}",
        loc="left",
        fontsize=14,
        pad=22,
    )
    ax.set_xlabel("Numeric benchmark metrics")
    ax.set_ylabel("Method")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label("signed delta vs reference (positive is better)")
    fig.tight_layout()

    out_path = out_dir / f"{fig_id}.PRELIMINARY.pdf"
    png_path = out_dir / f"{fig_id}.PRELIMINARY.png"
    fig.savefig(out_path)
    fig.savefig(png_path)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scrna-dir", type=Path,
                        default=REPO_ROOT / "results" / "reconciled" / "scrna")
    parser.add_argument("--scatac-dir", type=Path,
                        default=REPO_ROOT / "results" / "reconciled" / "scatac")
    parser.add_argument("--manifest", type=Path,
                        default=REPO_ROOT / "data" / "benchmark_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--partial-ok", action="store_true")
    parser.add_argument("--target-n", type=int, default=3,
                        help="Accepted for orchestrator parity.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    apply_publication_rcparams()

    if not args.manifest.exists():
        log.error("manifest missing: %s", args.manifest)
        return 1
    manifest = pd.read_csv(args.manifest)
    cases = _pick_cases(_available_result_index(args.scrna_dir, args.scatac_dir, manifest))
    if len(cases) < args.target_n and not args.partial_ok:
        log.error("only %d/%d case datasets — pass --partial-ok", len(cases), args.target_n)
        return 1
    if cases.empty:
        log.error("no candidate datasets")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_ids = ["fig10", "fig11", "fig12"]
    for fig_id, (_, row) in zip(fig_ids, cases.iterrows()):
        out = _render_case(row, fig_id=fig_id, out_dir=args.out_dir)
        log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
