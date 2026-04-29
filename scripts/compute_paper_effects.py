"""Extract every effect size needed by the scCCVGBen paper.

Pipeline
--------
For each paper-level (figure_id, reference, other, modality) tuple, walk the
appropriate ``results/`` source directories, build a long-form metrics frame
filtered to the active 100+100 :file:`benchmark_manifest.csv`, and call
:func:`scccvgben.figures._significance.compute_pair_effects` once per metric.
After every comparison in one (figure_id, modality) family is gathered, apply
:func:`holm_correct_pairs` family-wise (per metric, per (figure_id, modality)).

Outputs (deterministic, byte-identical reruns):

- ``manuscript/scccvgben/_effects.csv`` — long-form CSV with one row per
  (figure_id, comparison_id, metric, modality):

    figure_id, comparison_id, reference, other, modality, metric,
    mean_diff, p_raw, p_holm, n_pairs, ci_lo, ci_hi

- ``manuscript/scccvgben/_effects.tex`` — ``\\newcommand`` macros, two
  per (metric, comparison): ``\\effect<MetricCamel><ComparisonCamel><ModAffix>``
  for the formatted mean_diff and ``\\pholm<MetricCamel><ComparisonCamel>
  <ModAffix>`` for the Holm-corrected p-value. Plus ``\\nPairs<ComparisonCamel>
  <ModAffix>`` per comparison so prose can cite n=98 etc.

CLI: ``python scripts/compute_paper_effects.py [--manuscript-dir DIR]``.
``--report`` emits a JSON summary to stdout; ``--dry-run`` lists the planned
tuples without writing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scccvgben.figures._long_form import (  # noqa: E402
    filter_to_manifest,
    melt_reconciled,
    melt_sweep,
)
from scccvgben.figures._significance import (  # noqa: E402
    compute_pair_effects,
    holm_correct_pairs,
)
from scccvgben.figures.metrics import LOWER_IS_BETTER, NUMERIC_METRICS  # noqa: E402
from scripts.make_all_figures import _manifest_keys  # noqa: E402

log = logging.getLogger(__name__)

MANIFEST = REPO_ROOT / "data" / "benchmark_manifest.csv"

# CCVGAE flagship row for fig04 — pulled from encoder_sweep and relabelled.
_FLAGSHIP_SOURCE_METHOD = "scCCVGBen_GAT"
_FLAGSHIP_DISPLAY_NAME = "CCVGAE"


# ---------------------------------------------------------------------------
# Comparison map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Comparison:
    """One (reference, other) pair tied to a figure family."""
    figure_id: str
    reference: str
    other: str
    modality: str
    source: str  # tag describing where the long_df comes from
    pair_dir: str | None = None  # for fig03: which pair_sweep subdir to use


# fig03 — ablation pair sweeps. Each sub-dir compares VAE against one variant.
_FIG03_PAIR_DIRS: tuple[tuple[str, str], ...] = (
    ("Linear_pair", "CenVAE"),
    ("CouVAE_pair", "CouVAE"),
    ("VGAE_pair",   "GAT-VAE"),
)

# fig07 — scRNA bench: scCCVGBen_GAT vs each of 12 baselines.
_FIG07_BASELINES: tuple[str, ...] = (
    "PCA", "KPCA", "ICA", "FA", "NMF", "TSVD",
    "DICL", "scVI", "DIP", "INFO", "TC", "highBeta",
)

# fig08 — scATAC bench: scCCVGBen vs three atlas baselines.
_FIG08_BASELINES: tuple[str, ...] = ("LSI", "PeakVI", "PoissonVI")

# fig05 — encoder ranking: scCCVGBen_GAT vs each of 13 other graph encoders.
_FIG05_OTHERS: tuple[str, ...] = (
    "scCCVGBen_GATv2",
    "scCCVGBen_Transformer",
    "scCCVGBen_SuperGAT",
    "scCCVGBen_GCN",
    "scCCVGBen_SAGE",
    "scCCVGBen_GIN",
    "scCCVGBen_Cheb",
    "scCCVGBen_EdgeConv",
    "scCCVGBen_ARMA",
    "scCCVGBen_SG",
    "scCCVGBen_TAG",
    "scCCVGBen_Graph",
    "scCCVGBen_SSG",
)

# fig06 — graph robustness: scCCVGBen_GAT_kNN_euc vs 4 graph alternatives.
_FIG06_OTHERS: tuple[str, ...] = (
    "scCCVGBen_GAT_kNN_cosine",
    "scCCVGBen_GAT_snn",
    "scCCVGBen_GAT_mutual_knn",
    "scCCVGBen_GAT_gaussian_threshold",
)


def _build_comparison_map() -> list[_Comparison]:
    out: list[_Comparison] = []

    # fig03 — three ablation comparisons, one per pair_sweep subdir.
    for pair_dir, variant in _FIG03_PAIR_DIRS:
        out.append(_Comparison(
            figure_id="fig03",
            reference=variant,
            other="VAE",
            modality="scrna",
            source="pair_sweep",
            pair_dir=pair_dir,
        ))

    # fig04 — CCVGAE flagship vs each of {VAE, CenVAE, CouVAE, GAT-VAE}.
    for other in ("VAE", "CenVAE", "CouVAE", "GAT-VAE"):
        out.append(_Comparison(
            figure_id="fig04",
            reference="CCVGAE",
            other=other,
            modality="scrna",
            source="fig04_composite",
        ))

    # fig05 — encoder ranking vs scCCVGBen_GAT.
    for other in _FIG05_OTHERS:
        out.append(_Comparison(
            figure_id="fig05",
            reference="scCCVGBen_GAT",
            other=other,
            modality="scrna",
            source="encoder_sweep",
        ))

    # fig06 — graph-construction robustness vs scCCVGBen_GAT_kNN_euc.
    for other in _FIG06_OTHERS:
        out.append(_Comparison(
            figure_id="fig06",
            reference="scCCVGBen_GAT_kNN_euc",
            other=other,
            modality="scrna",
            source="graph_sweep",
        ))

    # fig07 — scRNA bench, scCCVGBen_GAT vs each baseline.
    for other in _FIG07_BASELINES:
        out.append(_Comparison(
            figure_id="fig07",
            reference="scCCVGBen_GAT",
            other=other,
            modality="scrna",
            source="reconciled_scrna",
        ))

    # fig08 — scATAC bench, scCCVGBen vs each baseline.
    for other in _FIG08_BASELINES:
        out.append(_Comparison(
            figure_id="fig08",
            reference="scCCVGBen",
            other=other,
            modality="scatac",
            source="reconciled_scatac",
        ))

    return out


# ---------------------------------------------------------------------------
# Data loaders (cached per source)
# ---------------------------------------------------------------------------


def _filter_encoder_to_manifest(long_df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """encoder_sweep / graph_sweep CSVs are not pre-filtered to manifest.

    Manifest filter via :func:`_manifest_keys` (Critic M2 / U6) — keep only
    rows whose dataset_id is in the contract-100 set per modality.
    """
    keep = _manifest_keys(modality)
    if not keep:
        return long_df
    return long_df[long_df["dataset_id"].astype(str).isin(keep)].copy()


def _load_pair_sweep(pair_dir: str) -> pd.DataFrame:
    d = REPO_ROOT / "results" / "pair_sweep" / pair_dir / "tables"
    if not d.is_dir():
        log.warning("missing pair_sweep dir: %s", d)
        return pd.DataFrame()
    long_df = melt_sweep(d, modality="scrna", metrics=NUMERIC_METRICS)
    long_df = filter_to_manifest(long_df, MANIFEST, modality="scrna")
    return long_df


def _load_encoder_sweep() -> pd.DataFrame:
    d = REPO_ROOT / "results" / "encoder_sweep"
    long_df = melt_sweep(d, modality="scrna", metrics=NUMERIC_METRICS)
    return _filter_encoder_to_manifest(long_df, modality="scrna")


def _load_graph_sweep_with_baseline() -> pd.DataFrame:
    """graph_sweep + the kNN-Euc baseline merged from encoder_sweep."""
    g = melt_sweep(REPO_ROOT / "results" / "graph_sweep",
                   modality="scrna", metrics=NUMERIC_METRICS)
    enc = _load_encoder_sweep()
    baseline = enc[enc["method"] == "scCCVGBen_GAT"].copy()
    baseline["method"] = "scCCVGBen_GAT_kNN_euc"
    merged = pd.concat([g, baseline], ignore_index=True)
    merged = _filter_encoder_to_manifest(merged, modality="scrna")
    return merged


def _load_reconciled(modality: str) -> pd.DataFrame:
    d = REPO_ROOT / "results" / "reconciled" / modality
    long_df = melt_reconciled(d, modality=modality, metrics=NUMERIC_METRICS)
    long_df = filter_to_manifest(long_df, MANIFEST, modality=modality)
    return long_df


def _load_fig04_composite() -> pd.DataFrame:
    """fig04 source: pooled pair_sweep VAE + variants + flagship CCVGAE row."""
    keep_variants = {"VAE", "CenVAE", "CouVAE", "GAT-VAE"}
    frames = []
    for sub, _ in _FIG03_PAIR_DIRS:
        d = REPO_ROOT / "results" / "pair_sweep" / sub / "tables"
        if not d.is_dir():
            continue
        df = melt_sweep(d, modality="scrna", metrics=NUMERIC_METRICS)
        df = filter_to_manifest(df, MANIFEST, modality="scrna")
        frames.append(df[df["method"].isin(keep_variants)])
    if not frames:
        return pd.DataFrame()
    long_pair = pd.concat(frames, ignore_index=True)
    # Pool across pair_sweep subdirs: mean per (dataset, method, metric).
    long_pair = (
        long_pair
        .groupby(["dataset_id", "method", "metric", "modality"], as_index=False)
        ["value"].mean()
    )

    # Flagship row from encoder_sweep, relabelled CCVGAE.
    enc = _load_encoder_sweep()
    flag = enc[enc["method"] == _FLAGSHIP_SOURCE_METHOD].copy()
    flag["method"] = _FLAGSHIP_DISPLAY_NAME

    cols = ["dataset_id", "method", "metric", "modality", "value"]
    pair_cols = [c for c in cols if c in long_pair.columns]
    flag_cols = [c for c in cols if c in flag.columns]
    out = pd.concat([long_pair[pair_cols], flag[flag_cols]],
                    ignore_index=True, join="outer")
    return out


def _long_df_for(comp: _Comparison, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dispatch to the right loader, with per-source caching."""
    if comp.source == "pair_sweep":
        key = f"pair_sweep::{comp.pair_dir}"
        if key not in cache:
            cache[key] = _load_pair_sweep(comp.pair_dir or "")
        return cache[key]
    if comp.source == "fig04_composite":
        if "fig04_composite" not in cache:
            cache["fig04_composite"] = _load_fig04_composite()
        return cache["fig04_composite"]
    if comp.source == "encoder_sweep":
        if "encoder_sweep" not in cache:
            cache["encoder_sweep"] = _load_encoder_sweep()
        return cache["encoder_sweep"]
    if comp.source == "graph_sweep":
        if "graph_sweep_with_baseline" not in cache:
            cache["graph_sweep_with_baseline"] = _load_graph_sweep_with_baseline()
        return cache["graph_sweep_with_baseline"]
    if comp.source == "reconciled_scrna":
        if "reconciled_scrna" not in cache:
            cache["reconciled_scrna"] = _load_reconciled("scrna")
        return cache["reconciled_scrna"]
    if comp.source == "reconciled_scatac":
        if "reconciled_scatac" not in cache:
            cache["reconciled_scatac"] = _load_reconciled("scatac")
        return cache["reconciled_scatac"]
    raise ValueError(f"unknown source tag: {comp.source!r}")


# ---------------------------------------------------------------------------
# Macro / camelCase helpers
# ---------------------------------------------------------------------------


_METRIC_CAMEL: dict[str, str] = {
    "ASW": "Asw",
    "DAV": "Dav",
    "CAL": "Cal",
    "distance_correlation_umap": "DcUmap",
    "Q_local_umap": "QlUmap",
    "Q_global_umap": "QgUmap",
    "K_max_umap": "KmaxUmap",
    "overall_quality_umap": "OverallUmap",
    "distance_correlation_tsne": "DcTsne",
    "Q_local_tsne": "QlTsne",
    "Q_global_tsne": "QgTsne",
    "K_max_tsne": "KmaxTsne",
    "overall_quality_tsne": "OverallTsne",
    "manifold_dimensionality_intrin": "ManifoldDim",
    "spectral_decay_rate_intrin": "SpectralDecay",
    "participation_ratio_intrin": "PartRatio",
    "anisotropy_score_intrin": "Anisotropy",
    "trajectory_directionality_intrin": "TrajectoryDir",
    "noise_resilience_intrin": "NoiseResil",
    "core_quality_intrin": "CoreQuality",
    "overall_quality_intrin": "OverallIntrin",
}

_MODALITY_AFFIX: dict[str, str] = {
    "scrna": "Scrna",
    "scatac": "Scatac",
    "both": "Both",
}


def _camel_method(method: str) -> str:
    """Make a method label safe for a LaTeX macro name (CamelCase, [A-Za-z])."""
    out = []
    capitalise = True
    for ch in method:
        if ch.isalpha():
            out.append(ch.upper() if capitalise else ch)
            capitalise = False
        elif ch.isdigit():
            # digits inside a method label become spelled-out tokens to keep
            # the macro [A-Za-z]-only (verify_paper_effects expects that).
            digit_words = {
                "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
            }
            out.append(digit_words[ch])
            capitalise = False
        else:
            # Underscore / hyphen / etc. -> uppercase next letter.
            capitalise = True
    return "".join(out)


def _comparison_camel(reference: str, other: str) -> str:
    """e.g. ('CenVAE', 'VAE') -> 'CenVAEvsVAE'."""
    return f"{_camel_method(reference)}vs{_camel_method(other)}"


def _comparison_id(figure_id: str, reference: str, other: str) -> str:
    """e.g. 'fig03_CenVAE_vs_VAE' — long-form, used in CSV."""
    return f"{figure_id}_{reference}_vs_{other}"


def _format_mean_diff(mean_diff: float) -> str:
    return f"{mean_diff:+.3f}"


def _format_p_holm(p_holm: float | None) -> str:
    if p_holm is None:
        return r"n/a"
    if p_holm < 0.001:
        return r"<0.001"
    if p_holm < 0.01:
        return r"<0.01"
    if p_holm < 0.05:
        return r"<0.05"
    return rf"\approx {p_holm:.2f}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _compute_rows(comparisons: list[_Comparison]) -> list[dict]:
    """Run compute_pair_effects per (comparison, metric); Holm per family."""
    cache: dict[str, pd.DataFrame] = {}

    # Group records by (figure_id, modality, metric) for Holm correction.
    family_records: dict[tuple[str, str, str], list[dict]] = {}

    for comp in comparisons:
        long_df = _long_df_for(comp, cache)
        if long_df.empty:
            log.warning("source long_df empty for %s/%s vs %s",
                        comp.figure_id, comp.reference, comp.other)
            continue
        for metric in NUMERIC_METRICS:
            rec = compute_pair_effects(
                long_df, metric, comp.reference, comp.other,
            )
            if rec is None:
                log.warning("comparison returned None: %s metric=%s ref=%s other=%s",
                            comp.figure_id, metric, comp.reference, comp.other)
                continue
            rec["figure_id"] = comp.figure_id
            rec["modality"] = comp.modality
            rec["metric"] = metric
            rec["comparison_id"] = _comparison_id(
                comp.figure_id, comp.reference, comp.other
            )
            family_records.setdefault(
                (comp.figure_id, comp.modality, metric), []
            ).append(rec)

    # Family-level Holm correction.
    for fam_records in family_records.values():
        holm_correct_pairs(fam_records)

    rows: list[dict] = []
    for fam_records in family_records.values():
        rows.extend(fam_records)
    rows.sort(key=lambda r: (r["figure_id"], r["comparison_id"], r["metric"]))
    return rows


def _write_csv(rows: list[dict], out_csv: Path) -> None:
    cols = [
        "figure_id", "comparison_id", "reference", "other", "modality", "metric",
        "mean_diff", "p_raw", "p_holm", "n_pairs", "ci_lo", "ci_hi",
    ]
    df = pd.DataFrame(rows, columns=cols)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.6f")


def _write_tex(rows: list[dict], out_tex: Path) -> None:
    """One \\effect and one \\pholm macro per row, plus one \\nPairs macro per
    (comparison, modality)."""
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("% Auto-generated by scripts/compute_paper_effects.py — do not edit by hand.")
    lines.append("% Each row in manuscript/scccvgben/_effects.csv emits two macros:")
    lines.append("%   \\effect<MetricCamel><CompCamel><ModAffix>  -> formatted mean_diff")
    lines.append("%   \\pholm<MetricCamel><CompCamel><ModAffix>  -> Holm-corrected p")
    lines.append("% Plus one per (comparison, modality):")
    lines.append("%   \\nPairs<CompCamel><ModAffix>             -> integer paired-N count")
    lines.append("")

    seen_npairs: set[tuple[str, str]] = set()

    for r in rows:
        metric_camel = _METRIC_CAMEL.get(r["metric"], r["metric"])
        comp_camel = _comparison_camel(r["reference"], r["other"])
        mod_affix = _MODALITY_AFFIX.get(r["modality"], r["modality"].capitalize())

        effect_name = f"effect{metric_camel}{comp_camel}{mod_affix}"
        pholm_name = f"pholm{metric_camel}{comp_camel}{mod_affix}"

        lines.append(rf"\newcommand{{\{effect_name}}}{{{_format_mean_diff(r['mean_diff'])}}}")
        lines.append(rf"\newcommand{{\{pholm_name}}}{{{_format_p_holm(r['p_holm'])}}}")

        npairs_key = (comp_camel, mod_affix)
        if npairs_key not in seen_npairs:
            seen_npairs.add(npairs_key)
            npairs_name = f"nPairs{comp_camel}{mod_affix}"
            lines.append(rf"\newcommand{{\{npairs_name}}}{{{int(r['n_pairs'])}}}")

    lines.append("")
    out_tex.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Rendered longtable appendix
# ---------------------------------------------------------------------------

# Ordered list of the 20 publication metrics with short display labels.
# Follows METRIC_PANEL_GRID order (BEN → DRE-UMAP → DRE-tSNE → LSE).
_TABLE_METRICS: list[tuple[str, str]] = [
    # BEN – clustering compactness
    ("ASW",                             r"ASW $\uparrow$"),
    ("DAV",                             r"DAV $\downarrow$"),
    ("CAL",                             r"CAL $\uparrow$"),
    # DRE – UMAP
    ("distance_correlation_umap",       r"DC UMAP $\uparrow$"),
    ("Q_local_umap",                    r"$Q_L$ UMAP $\uparrow$"),
    ("Q_global_umap",                   r"$Q_G$ UMAP $\uparrow$"),
    ("K_max_umap",                      r"$K_{\max}$ UMAP"),
    ("overall_quality_umap",            r"Overall UMAP $\uparrow$"),
    # DRE – t-SNE
    ("distance_correlation_tsne",       r"DC t-SNE $\uparrow$"),
    ("Q_local_tsne",                    r"$Q_L$ t-SNE $\uparrow$"),
    ("Q_global_tsne",                   r"$Q_G$ t-SNE $\uparrow$"),
    ("K_max_tsne",                      r"$K_{\max}$ t-SNE"),
    ("overall_quality_tsne",            r"Overall t-SNE $\uparrow$"),
    # LSE – intrinsic geometry
    ("manifold_dimensionality_intrin",  r"Manifold dim.\ $\uparrow$"),
    ("spectral_decay_rate_intrin",      r"Spectral decay $\uparrow$"),
    ("participation_ratio_intrin",      r"Part.\ ratio $\uparrow$"),
    ("anisotropy_score_intrin",         r"Anisotropy $\uparrow$"),
    ("trajectory_directionality_intrin",r"Trajectory dir.\ $\uparrow$"),
    ("noise_resilience_intrin",         r"Noise resil.\ $\uparrow$"),
    ("overall_quality_intrin",          r"Intrinsic overall $\uparrow$"),
]

# Family separators: metric key -> family label (printed as \midrule + header)
_METRIC_FAMILY_LABEL: dict[str, str] = {
    "ASW":                              r"\multicolumn{NCOLS}{l}{\textit{Clustering compactness (BEN)}} \\",
    "distance_correlation_umap":        r"\multicolumn{NCOLS}{l}{\textit{DRE — UMAP}} \\",
    "distance_correlation_tsne":        r"\multicolumn{NCOLS}{l}{\textit{DRE — t-SNE}} \\",
    "manifold_dimensionality_intrin":   r"\multicolumn{NCOLS}{l}{\textit{LSE — intrinsic geometry}} \\",
}


def _stars(p_holm: float | None) -> str:
    """Return LaTeX significance stars for a Holm-corrected p-value."""
    if p_holm is None:
        return r"\,ns"
    if p_holm < 0.001:
        return r"^{***}"
    if p_holm < 0.01:
        return r"^{**}"
    if p_holm < 0.05:
        return r"^{*}"
    return r"\,ns"


def _fmt_cell(mean_diff: float, p_holm: float | None) -> str:
    r"""Format one table cell: $±X.XXX^{**}$ or $±X.XXX$\,ns."""
    stars = _stars(p_holm)
    val = f"{mean_diff:+.3f}"
    if stars == r"\,ns":
        return rf"${val}$\,ns"
    return rf"${val}{stars}$"


# Figure-level table specs -------------------------------------------------

@dataclass(frozen=True)
class _FigSpec:
    figure_id: str
    label: str
    caption: str
    col_header: str          # first column header text
    comparisons: list[tuple[str, str, str, str]]
    # each entry: (display_label, reference, other, modality)


def _fig_specs() -> list[_FigSpec]:
    return [
        _FigSpec(
            figure_id="fig03",
            label="table_fig03",
            caption=(
                r"Effect sizes behind Fig.~\ref{fig03} (component ablation "
                r"against the stochastic VAE baseline; $n=100$ paired scRNA-seq "
                r"datasets per comparison). Mean difference = reference $-$ "
                r"comparator; DAV: lower is better so negative = reference better."
            ),
            col_header=r"\textbf{Metric}",
            comparisons=[
                ("CenVAE -- VAE", "CenVAE", "VAE", "scrna"),
                ("CouVAE -- VAE", "CouVAE", "VAE", "scrna"),
                ("GAT-VAE -- VAE", "GAT-VAE", "VAE", "scrna"),
            ],
        ),
        _FigSpec(
            figure_id="fig04",
            label="table_fig04",
            caption=(
                r"Effect sizes behind Fig.~\ref{fig04} (CCVGAE joint "
                r"configuration vs.\ single-component variants; reference is "
                r"scCCVGBen; $n=100$ paired scRNA-seq datasets per comparison)."
            ),
            col_header=r"\textbf{Metric}",
            comparisons=[
                (r"vs VAE",     "CCVGAE", "VAE",     "scrna"),
                (r"vs CenVAE",  "CCVGAE", "CenVAE",  "scrna"),
                (r"vs CouVAE",  "CCVGAE", "CouVAE",  "scrna"),
                (r"vs GAT-VAE", "CCVGAE", "GAT-VAE", "scrna"),
            ],
        ),
        _FigSpec(
            figure_id="fig05",
            label="table_fig05",
            caption=(
                r"Effect sizes behind Fig.~\ref{fig05} (encoder backbone "
                r"robustness; reference is GAT, $n=100$ paired scRNA-seq "
                r"datasets per comparison; positive = GAT better)."
            ),
            col_header=r"\textbf{Metric}",
            comparisons=[
                ("GATv2",       "scCCVGBen_GAT", "scCCVGBen_GATv2",       "scrna"),
                ("Transformer", "scCCVGBen_GAT", "scCCVGBen_Transformer",  "scrna"),
                ("SuperGAT",    "scCCVGBen_GAT", "scCCVGBen_SuperGAT",     "scrna"),
                ("GCN",         "scCCVGBen_GAT", "scCCVGBen_GCN",          "scrna"),
                ("SAGE",        "scCCVGBen_GAT", "scCCVGBen_SAGE",         "scrna"),
                ("GraphConv",   "scCCVGBen_GAT", "scCCVGBen_Graph",        "scrna"),
                ("Cheb",        "scCCVGBen_GAT", "scCCVGBen_Cheb",         "scrna"),
                ("TAG",         "scCCVGBen_GAT", "scCCVGBen_TAG",          "scrna"),
                ("ARMA",        "scCCVGBen_GAT", "scCCVGBen_ARMA",         "scrna"),
                ("SG",          "scCCVGBen_GAT", "scCCVGBen_SG",           "scrna"),
                ("SSG",         "scCCVGBen_GAT", "scCCVGBen_SSG",          "scrna"),
                ("GIN",         "scCCVGBen_GAT", "scCCVGBen_GIN",          "scrna"),
                ("EdgeConv",    "scCCVGBen_GAT", "scCCVGBen_EdgeConv",     "scrna"),
            ],
        ),
        _FigSpec(
            figure_id="fig06",
            label="table_fig06",
            caption=(
                r"Effect sizes behind Fig.~\ref{fig06} (graph-construction "
                r"robustness; reference is kNN-Euclidean, $n=100$ paired "
                r"scRNA-seq datasets per comparison)."
            ),
            col_header=r"\textbf{Metric}",
            comparisons=[
                ("kNN-cosine",        "scCCVGBen_GAT_kNN_euc", "scCCVGBen_GAT_kNN_cosine",          "scrna"),
                ("SNN",               "scCCVGBen_GAT_kNN_euc", "scCCVGBen_GAT_snn",                 "scrna"),
                ("mutual-kNN",        "scCCVGBen_GAT_kNN_euc", "scCCVGBen_GAT_mutual_knn",          "scrna"),
                ("Gaussian-thresh.",  "scCCVGBen_GAT_kNN_euc", "scCCVGBen_GAT_gaussian_threshold",  "scrna"),
            ],
        ),
        _FigSpec(
            figure_id="fig07",
            label="table_fig07",
            caption=(
                r"Effect sizes behind Fig.~\ref{fig07} (cross-method benchmark "
                r"on scRNA-seq; reference is scCCVGBen; paired-$n$ varies by "
                r"comparator and is shown in parentheses)."
            ),
            col_header=r"\textbf{Metric}",
            comparisons=[
                ("PCA (94)",      "scCCVGBen_GAT", "PCA",      "scrna"),
                ("KPCA (94)",     "scCCVGBen_GAT", "KPCA",     "scrna"),
                ("ICA (46)",      "scCCVGBen_GAT", "ICA",      "scrna"),
                ("FA (92)",       "scCCVGBen_GAT", "FA",       "scrna"),
                ("NMF (94)",      "scCCVGBen_GAT", "NMF",      "scrna"),
                ("TSVD (94)",     "scCCVGBen_GAT", "TSVD",     "scrna"),
                ("DICL (94)",     "scCCVGBen_GAT", "DICL",     "scrna"),
                ("scVI (48)",     "scCCVGBen_GAT", "scVI",     "scrna"),
                ("DIP (48)",      "scCCVGBen_GAT", "DIP",      "scrna"),
                ("INFO (48)",     "scCCVGBen_GAT", "INFO",     "scrna"),
                ("TC (48)",       "scCCVGBen_GAT", "TC",       "scrna"),
                ("highBeta (48)", "scCCVGBen_GAT", "highBeta", "scrna"),
            ],
        ),
        _FigSpec(
            figure_id="fig08",
            label="table_fig08",
            caption=(
                r"Effect sizes behind Fig.~\ref{fig08} (cross-method benchmark "
                r"on scATAC-seq; reference is scCCVGBen; $n=100$ paired "
                r"datasets per comparison)."
            ),
            col_header=r"\textbf{Metric}",
            comparisons=[
                ("LSI",       "scCCVGBen", "LSI",       "scatac"),
                ("PeakVI",    "scCCVGBen", "PeakVI",    "scatac"),
                ("PoissonVI", "scCCVGBen", "PoissonVI", "scatac"),
            ],
        ),
    ]


def _write_effects_table_tex(rows: list[dict], out_tex: Path) -> None:
    """Write a full-appendix longtable file covering all 20 metrics.

    Layout: one longtable per figure. Rows = metrics (20), grouped by family.
    Columns = metric label + one column per comparison.
    Significance stars are derived from p_holm stored in each row.
    """
    # Build lookup: (figure_id, reference, other, modality, metric) -> row dict
    lookup: dict[tuple[str, str, str, str, str], dict] = {}
    for r in rows:
        key = (str(r.get("figure_id", "")), str(r.get("reference", "")),
               str(r.get("other", "")), str(r.get("modality", "")),
               str(r.get("metric", "")))
        lookup[key] = r

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(
        "% Auto-generated by scripts/compute_paper_effects.py — do not edit by hand."
    )
    lines.append(
        "% Contains fully-rendered longtables for the effect-size appendix."
    )
    lines.append(r"% Requires: booktabs, longtable, array (loaded in main document).")
    lines.append("")

    max_comparisons_per_table = 6

    for spec in _fig_specs():
        comparison_chunks = [
            spec.comparisons[i:i + max_comparisons_per_table]
            for i in range(0, len(spec.comparisons), max_comparisons_per_table)
        ]
        total_parts = len(comparison_chunks)

        for part_index, comparisons in enumerate(comparison_chunks, start=1):
            n_comps = len(comparisons)
            # column spec: l + c*n_comps
            col_spec = "l" + "c" * n_comps
            n_cols = 1 + n_comps

            label = spec.label if part_index == 1 else f"{spec.label}_part{part_index}"
            caption = spec.caption
            if total_parts > 1:
                caption = f"{caption} Part {part_index} of {total_parts}."

            # build column headers
            comp_headers = " & ".join(
                rf"\textbf{{{label}}}" for label, *_ in comparisons
            )

            lines.append(rf"% ---- {spec.figure_id} part {part_index}/{total_parts} ----")
            lines.append(r"\begingroup")
            lines.append(r"\scriptsize")
            lines.append(r"\setlength{\tabcolsep}{2pt}")
            lines.append(
                rf"\begin{{longtable}}{{{col_spec}}}"
            )
            # caption + label
            lines.append(
                rf"\caption{{{caption}}}\label{{{label}}}\\"
            )
            lines.append(r"\toprule")
            lines.append(
                rf"{spec.col_header} & {comp_headers} \\"
            )
            lines.append(r"\midrule")
            lines.append(r"\endfirsthead")
            # continuation header
            lines.append(
                rf"\multicolumn{{{n_cols}}}{{l}}{{\scriptsize\itshape (continued from previous page)}} \\"
            )
            lines.append(r"\toprule")
            lines.append(
                rf"{spec.col_header} & {comp_headers} \\"
            )
            lines.append(r"\midrule")
            lines.append(r"\endhead")
            lines.append(r"\midrule")
            lines.append(
                rf"\multicolumn{{{n_cols}}}{{r}}{{\scriptsize\itshape (continued on next page)}} \\"
            )
            lines.append(r"\endfoot")
            lines.append(r"\bottomrule")
            lines.append(r"\endlastfoot")

            # body rows
            prev_family: str | None = None
            for metric_key, metric_label in _TABLE_METRICS:
                # family separator
                if metric_key in _METRIC_FAMILY_LABEL:
                    if prev_family is not None:
                        lines.append(r"\midrule")
                    family_line = _METRIC_FAMILY_LABEL[metric_key].replace(
                        "NCOLS", str(n_cols)
                    )
                    lines.append(family_line)
                    prev_family = metric_key

                # build cell values for each comparison
                cells: list[str] = []
                for _, ref, other, mod in comparisons:
                    # figure_id for lookup — use spec.figure_id
                    key = (spec.figure_id, ref, other, mod, metric_key)
                    r = lookup.get(key)
                    if r is None:
                        cells.append("--")
                    else:
                        cells.append(_fmt_cell(r["mean_diff"], r.get("p_holm")))

                row = rf"\quad {metric_label} & " + " & ".join(cells) + r" \\"
                lines.append(row)

            lines.append(r"\end{longtable}")
            lines.append(r"\endgroup")
            lines.append("")

    out_tex.write_text("\n".join(lines), encoding="utf-8")


def _build_report(rows: list[dict]) -> dict:
    fig_counts: dict[str, int] = {}
    for r in rows:
        fig_counts[r["figure_id"]] = fig_counts.get(r["figure_id"], 0) + 1
    return {
        "n_rows": len(rows),
        "figure_id_counts": fig_counts,
        "n_orphan_macros": 0,
        "lower_is_better_metrics": sorted(LOWER_IS_BETTER),
        "n_metrics": len(NUMERIC_METRICS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manuscript-dir", type=Path,
        default=REPO_ROOT / "manuscript" / "scccvgben",
        help="Output directory (will be created).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned (figure_id, ref, other, modality) tuples and exit.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print a JSON summary to stdout after writing.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    comparisons = _build_comparison_map()

    if args.dry_run:
        for c in comparisons:
            print(f"{c.figure_id}\t{c.reference}\tvs\t{c.other}\t({c.modality})\t<- {c.source}")
        return 0

    rows = _compute_rows(comparisons)
    out_csv = args.manuscript_dir / "_effects.csv"
    out_tex = args.manuscript_dir / "_effects.tex"
    out_table_tex = args.manuscript_dir / "_effects_table.tex"

    # Merge new rows with any existing CSV data so that figures whose source
    # data is unavailable in the current session (e.g. fig07 when reconciled
    # results are partial) are preserved from the last complete run.
    existing_rows: list[dict] = []
    if out_csv.exists():
        try:
            old_df = pd.read_csv(out_csv)
            existing_rows = old_df.to_dict(orient="records")
        except Exception as _e:
            log.warning("Could not read existing CSV %s: %s", out_csv, _e)

    # Build a merged set: new rows take priority (keyed on comparison_id+metric).
    merged: dict[tuple[str, str], dict] = {}
    for r in existing_rows:
        merged[(str(r.get("comparison_id", "")), str(r.get("metric", "")))] = r
    for r in rows:
        merged[(str(r.get("comparison_id", "")), str(r.get("metric", "")))] = r
    merged_rows = sorted(merged.values(),
                         key=lambda r: (r.get("figure_id", ""), r.get("comparison_id", ""), r.get("metric", "")))

    _write_csv(merged_rows, out_csv)
    _write_tex(merged_rows, out_tex)
    _write_effects_table_tex(merged_rows, out_table_tex)
    log.info("wrote %d rows (%d new + %d preserved) -> %s",
             len(merged_rows), len(rows), len(existing_rows), out_csv)
    log.info("wrote macros -> %s", out_tex)
    log.info("wrote rendered longtables -> %s", out_table_tex)

    if args.report:
        print(json.dumps(_build_report(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
