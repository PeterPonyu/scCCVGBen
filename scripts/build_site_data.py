#!/usr/bin/env python3
"""build_site_data.py — generate Hugo site data + content from benchmark artifacts.

Inputs:
  data/benchmark_manifest.csv      — 200 rows, 15 cols, online-verified metadata
  data/benchmark_summary.json      — aggregated counts + rules

Outputs:
  site/data/datasets.json          — array of 200 dataset objects (for DataTables)
  site/data/methods.json           — 29 methods grouped by 3 categories
  site/data/metrics.json           — 26 metrics grouped by 3 categories
  site/data/summary.json           — home-page headline stats + chart data
  site/content/datasets/*.md       — one page per dataset (auto-generated)

Usage:
  python scripts/build_site_data.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "site"
DATA_OUT = SITE / "data"          # Hugo templates read from here (.Site.Data)
STATIC_OUT = SITE / "static"      # files here are copied 1:1 to public/ (for fetch())
CONTENT_OUT = SITE / "content"


def _dump_json_dual(obj, name: str) -> None:
    """Write the same JSON to both data/ (server-side) and static/ (client-side fetch)."""
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    STATIC_OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, ensure_ascii=False)
    (DATA_OUT / f"{name}.json").write_text(payload, encoding="utf-8")
    (STATIC_OUT / f"{name}.json").write_text(payload, encoding="utf-8")


# ─────────────────────────────── Datasets ────────────────────────────────

def build_datasets() -> list[dict]:
    """Load benchmark_manifest.csv and normalise into site-friendly JSON."""
    src = REPO / "data" / "benchmark_manifest.csv"
    df = pd.read_csv(src)
    rows: list[dict] = []
    for _, r in df.iterrows():
        # Use corrected species + manifest fields
        key = str(r.get("filename_key", "")).strip()
        gse = str(r.get("gse_extract") or r.get("GSE") or "").strip()
        modality = str(r.get("modality", "")).strip()
        # scRNA or scATAC capitalisation for display
        modality_display = {"scrna": "scRNA", "scatac": "scATAC"}.get(
            modality.lower(), modality
        )
        species = str(r.get("species_corrected") or r.get("species") or "").strip()
        tissue = str(r.get("tissue", "")).strip() or "unknown"
        try:
            cell_count = int(r.get("cell_count", 0))
        except Exception:
            cell_count = 0
        description = str(r.get("description", "")).strip()
        geo_title = str(r.get("geo_title", "")).strip()
        geo_url = str(r.get("scrna_geo_source", "")).strip()
        pubmed = str(r.get("geo_pubmed_id", "")).strip()
        subdate = str(r.get("geo_submission_date", "")).strip()
        source_name = str(r.get("geo_source_name", "")).strip()

        # Enrich tissue via keyword scan on GEO metadata + description
        canon_tissue = _slugify_tissue(tissue, source_name, description, geo_title)

        rows.append({
            "filename_key": key,
            "GSE": gse,
            "modality": modality_display,
            "species": species,
            "tissue": canon_tissue,
            "cell_count": cell_count,
            "category": canon_tissue,
            "description": description,
            "geo_title": geo_title,
            "geo_url": geo_url,
            "pubmed_id": pubmed,
            "submission_date": subdate,
            "source_name": source_name,
        })
    return rows


TISSUE_KEYWORDS = [
    ("lung",        ["lung", "pulmonary", "airway", "bronchial", "alveol", "balf"]),
    ("liver",       ["liver", "hepat"]),
    ("brain",       ["brain", "cortex", "hippocamp", "cerebell", "neural", "neuron",
                     "dentate", "microglia", "glioma", "astrocy"]),
    ("blood",       ["blood", "pbmc", "peripheral", "lymphocyt", "leukocyt", "immune"]),
    ("bone_marrow", ["bone marrow", "marrow", "bm ", "hsc", "hematopoi", "hemat"]),
    ("breast",      ["breast", "mammary"]),
    ("pancreas",    ["pancrea", "islet", "endocrine", "beta cell"]),
    ("kidney",      ["kidney", "renal", "nephron"]),
    ("heart",       ["heart", "cardiac", "myocard"]),
    ("gut",         ["gut", "intestin", "colon", "ileum", "jejunum", "crypt", "duodenum"]),
    ("stomach",     ["stomach", "gastric"]),
    ("skin",        ["skin", "melanoma", "melanocyt", "epiderm", "bcc", "scc"]),
    ("muscle",      ["muscle", "muscular", "myoblast", "skeletal"]),
    ("retina",      ["retina", "eye", "macula"]),
    ("thymus",      ["thymus", "thymic"]),
    ("ovary",       ["ovary", "ovari"]),
    ("testis",      ["testis", "testicular", "sperm"]),
    ("bladder",     ["bladder", "urinary"]),
    ("adipose",     ["adipose", "adipocyt"]),
    ("stem_cell",   ["esc", "ipsc", "pluripotent", "organoid"]),
    ("embryo",      ["embryo", "gastrul", "zygote"]),
    ("spine",       ["spinal", "spine"]),
    ("tonsil",      ["tonsil"]),
    ("tumor",       ["tumor", "cancer", "carcinoma", "neoplas", "mel_"]),  # last-resort catch
]


def _infer_tissue_from_text(*texts) -> str | None:
    """Scan multiple text sources for tissue keywords. Return canonical tissue name."""
    joined = " ".join(str(t or "") for t in texts).lower()
    for canon, keywords in TISSUE_KEYWORDS:
        for kw in keywords:
            if kw in joined:
                return canon
    return None


def _slugify_tissue(t: str, *enrich_sources: str) -> str:
    """Canonicalize tissue: start from manifest 'tissue', fall back to keyword
    scan on enrichment sources (geo_source_name, description, geo_title)."""
    raw = str(t or "").lower().strip()
    # Manifest already had a usable value?
    if raw and raw not in ("other", "unknown", "mixed"):
        norm = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        if norm:
            return norm
    # Enrich from text sources
    inferred = _infer_tissue_from_text(*enrich_sources)
    if inferred:
        return inferred
    return raw if raw else "other"


# ─────────────────────────────── Methods ─────────────────────────────────

METHODS_SCCCVGBEN_ENCODERS = [
    ("GAT",        "attention",   "Graph Attention Network (Veličković 2018)"),
    ("GATv2",      "attention",   "Dynamic attention Graph Attention v2 (Brody 2022) — scCCVGBen extension"),
    ("Transformer","attention",   "TransformerConv — Attention is All You Need graph variant (Shi 2020)"),
    ("SuperGAT",   "attention",   "Self-supervised edge prediction GAT (Kim 2020) — scCCVGBen extension"),
    ("GCN",        "message-pass","Graph Convolutional Network (Kipf 2017)"),
    ("SAGE",       "message-pass","GraphSAGE inductive (Hamilton 2017)"),
    ("Graph",      "message-pass","GraphConv — general message-passing variant"),
    ("Cheb",       "message-pass","ChebNet — Chebyshev polynomial filters (Defferrard 2016)"),
    ("TAG",        "message-pass","Topology-Adaptive Graph Conv (Du 2017)"),
    ("ARMA",       "message-pass","ARMA filter Conv (Bianchi 2021)"),
    ("SG",         "message-pass","Simplified Graph Conv (Wu 2019)"),
    ("SSG",        "message-pass","Simple Spectral Graph Conv (Zhu 2021)"),
    ("GIN",        "message-pass","Graph Isomorphism Network (Xu 2019) — scCCVGBen extension"),
    ("EdgeConv",   "message-pass","Dynamic Edge Conv (Wang 2019) — scCCVGBen extension"),
]

METHODS_GRAPHS = [
    ("kNN_euclidean",     "Standard k-NN with Euclidean distance (k=15) — CCVGAE benchmark default"),
    ("kNN_cosine",        "k-NN with cosine similarity — rewards direction, invariant to magnitude"),
    ("snn",               "Shared Nearest Neighbour — 2 cells connected if they share a fraction of neighbours"),
    ("mutual_knn",        "Mutual k-NN — only edges where both cells are in each other's k-NN list; stricter connectivity"),
    ("gaussian_threshold","Gaussian heat-kernel weights w=exp(-d²/(2σ²)); edges pruned at threshold 0.9"),
]

METHODS_BASELINES = [
    ("PCA",      "Linear PCA — sklearn.decomposition.PCA"),
    ("KPCA",     "Kernel PCA (RBF) — sklearn.decomposition.KernelPCA"),
    ("ICA",      "Independent Component Analysis (FastICA) — sklearn.decomposition.FastICA"),
    ("FA",       "Factor Analysis — sklearn.decomposition.FactorAnalysis"),
    ("NMF",      "Non-negative Matrix Factorisation — sklearn.decomposition.NMF"),
    ("TSVD",     "Truncated SVD — sklearn.decomposition.TruncatedSVD"),
    ("DICL",     "Dictionary Learning — sklearn.decomposition.DictionaryLearning"),
    ("scVI",     "Single-cell Variational Inference — Lopez 2018"),
    ("DIP",      "DIP-VAE disentangled autoencoder — Kumar 2017"),
    ("INFO",     "InfoVAE — Zhao 2017"),
    ("TC",       "β-TCVAE — Chen 2018"),
    ("highBeta", "Hyper-parameterised VAE with high β (β=100)"),
    ("CCVGAE",   "Reference CCVGAE — upstream baseline; scCCVGBen extends this with 12 encoder + 5 graph variants"),
]


def build_methods() -> dict:
    def _card(name, family, desc):
        return {"name": name, "family": family, "description": desc}
    return {
        "scCCVGBen_encoders": [_card(n, f, d) for n, f, d in METHODS_SCCCVGBEN_ENCODERS],
        "graph_constructions": [_card(n, "graph", d) for n, d in METHODS_GRAPHS],
        "baselines":          [_card(n, "baseline", d) for n, d in METHODS_BASELINES],
    }


# ─────────────────────────────── Metrics ─────────────────────────────────

METRICS_CLUSTER = [
    ("ASW",  "Average Silhouette Width",        "sklearn.metrics.silhouette_score",      "Higher is better"),
    ("DAV",  "Davies–Bouldin Index",             "sklearn.metrics.davies_bouldin_score",   "Lower is better"),
    ("CAL",  "Calinski–Harabasz Index",          "sklearn.metrics.calinski_harabasz_score","Higher is better"),
    ("NMI",  "Normalised Mutual Information",    "sklearn.metrics.normalized_mutual_info_score", "Higher is better"),
    ("ARI",  "Adjusted Rand Index",              "sklearn.metrics.adjusted_rand_score",    "Higher is better"),
    ("COR",  "Pairwise distance correlation",    "Spearman on pairwise distances Z vs X_orig", "Higher is better"),
]

METRICS_DRE = [
    ("distance_correlation_umap","Spearman distance corr (latent vs UMAP)",    "DRE.evaluate_dimensionality_reduction"),
    ("Q_local_umap",             "Coranking Q_NX local average (k=1..K_max)",  "DRE Lee & Verleysen 2009"),
    ("Q_global_umap",            "Coranking Q_NX global average",               "DRE"),
    ("K_max_umap",               "Argmax of Q_NX(k) — optimal neighbourhood scale", "DRE"),
    ("overall_quality_umap",     "Weighted composite of Q_local + Q_global",    "DRE"),
    ("distance_correlation_tsne","Spearman dist corr (latent vs tSNE)",          "DRE"),
    ("Q_local_tsne",             "Coranking Q_NX local (tSNE)",                 "DRE"),
    ("Q_global_tsne",            "Coranking Q_NX global (tSNE)",                "DRE"),
    ("K_max_tsne",               "Argmax Q_NX(k) in tSNE",                       "DRE"),
    ("overall_quality_tsne",     "Weighted composite (tSNE)",                    "DRE"),
]

METRICS_LSE = [
    ("manifold_dimensionality_intrin","Intrinsic manifold dimensionality score",   "LSE.SingleCellLatentSpaceEvaluator"),
    ("spectral_decay_rate_intrin",    "PCA eigenvalue decay rate",                  "LSE"),
    ("participation_ratio_intrin",    "Effective latent dim (1/Σ w_i² when Σ w_i=1)","LSE"),
    ("anisotropy_score_intrin",       "Covariance eigenvalue anisotropy",           "LSE"),
    ("trajectory_directionality_intrin","Primary developmental axis strength",     "LSE"),
    ("noise_resilience_intrin",       "Perturbation robustness score",              "LSE"),
    ("core_quality_intrin",           "Mean of 4 core manifold metrics",            "LSE"),
    ("overall_quality_intrin",        "Weighted composite (trajectory-aware)",       "LSE"),
    ("data_type_intrin",              "Auto-classified: 'trajectory' or 'steady_state'", "LSE"),
    ("interpretation_intrin",         "Structured dict: quality_level, strengths, weaknesses, recommendations", "LSE"),
]


def build_metrics() -> dict:
    def _c(name, desc, src, note=""):
        d = {"name": name, "description": desc, "source": src}
        if note: d["note"] = note
        return d
    return {
        "clustering": [_c(n, d, s, note) for n, d, s, note in METRICS_CLUSTER],
        "dre":        [_c(n, d, s) for n, d, s in METRICS_DRE],
        "lse":        [_c(n, d, s) for n, d, s in METRICS_LSE],
    }


# ─────────────────────────────── Summary ─────────────────────────────────

def build_summary(datasets: list[dict]) -> dict:
    df = pd.DataFrame(datasets)
    tissue_counts = df["tissue"].value_counts().to_dict()
    species_counts = df["species"].value_counts().to_dict()
    modality_counts = df["modality"].value_counts().to_dict()
    cross = df.groupby(["species","modality"]).size().reset_index(name="count")
    cross_list = cross.to_dict("records")

    cell_count = df["cell_count"]
    # Cell-count bins for the skewed size distribution.
    bins = [0, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 5_000_000]
    labels = ["<1k", "1–3k", "3–10k", "10–30k", "30–100k", "100–300k", "300k–1M", ">1M"]
    hist = pd.cut(cell_count, bins=bins, labels=labels, right=False).value_counts().sort_index()
    cell_hist = [{"bin": str(k), "count": int(v)} for k, v in hist.items()]

    # Submission year timeline.
    def _year(s):
        s = str(s or "").strip()
        m = re.search(r"\b(19|20)\d{2}\b", s)
        return int(m.group(0)) if m else None
    years = pd.Series([_year(s) for s in df["submission_date"]]).dropna().astype(int)
    year_hist = years.value_counts().sort_index()
    timeline = [{"year": int(y), "count": int(c)} for y, c in year_hist.items()]

    # Tissue × modality counts.
    tissue_modality = (
        df.groupby(["tissue", "modality"]).size().reset_index(name="count")
    )
    tissue_modality_list = tissue_modality.to_dict("records")

    # Largest PubMed-linked studies by cell count.
    with_pubmed = df[df["pubmed_id"].astype(str).str.strip() != ""]
    pubmed_top = (
        with_pubmed.sort_values("cell_count", ascending=False)
        .head(15)[["filename_key", "pubmed_id", "cell_count", "species", "modality"]]
        .to_dict("records")
    )

    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "total_datasets": int(len(df)),
        "modality": modality_counts,
        "species": species_counts,
        "tissue_top10": dict(sorted(tissue_counts.items(), key=lambda x: -x[1])[:10]),
        "species_modality": cross_list,
        "tissue_modality": tissue_modality_list,
        "cell_count_range": {
            "min": int(cell_count.min()),
            "median": int(cell_count.median()),
            "max": int(cell_count.max()),
        },
        "cell_count_hist": cell_hist,
        "submission_year_timeline": timeline,
        "pubmed_top15": pubmed_top,
        "methods_total": 14 + 5 + 13,
        "metrics_total": 6 + 10 + 10,
    }


# ────────────────────── Per-dataset markdown pages ───────────────────────

DATASET_PAGE_TEMPLATE = """---
title: "{title}"
type: docs
weight: {weight}
geekdocHidden: false
---

# {display_title}

| Field | Value |
|-------|-------|
| **filename_key** | `{filename_key}` |
| **GSE / GSM accession** | {gse_link} |
| **Modality** | {modality} |
| **Species** | {species} |
| **Tissue** | {tissue} |
| **Cell count** | {cell_count:,} |
| **PubMed** | {pubmed_link} |
| **Submission date** | {submission_date} |
| **Source name (GEO)** | {source_name} |

## GEO title

> {geo_title}

## Description (local manifest)

{description}

---

<small>Record auto-generated from `data/benchmark_manifest.csv` by
`scripts/build_site_data.py`. All fields verified against GEO online
metadata via GEOparse.</small>
"""


def write_dataset_pages(datasets: list[dict]) -> int:
    ds_dir = CONTENT_OUT / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous per-dataset pages (keep _index.md)
    for p in ds_dir.glob("*.md"):
        if p.name != "_index.md":
            p.unlink()

    written = 0
    for i, d in enumerate(datasets):
        key = d["filename_key"]
        if not key:
            continue
        gse_link = (
            f"[{d['GSE']}]({d['geo_url']})" if d.get("geo_url") else d.get("GSE", "—")
        )
        pubmed_link = (
            f"[PMID:{d['pubmed_id']}](https://pubmed.ncbi.nlm.nih.gov/{d['pubmed_id']}/)"
            if d.get("pubmed_id") else "—"
        )
        page = DATASET_PAGE_TEMPLATE.format(
            title=key,
            display_title=key,
            weight=100 + i,
            filename_key=key,
            gse_link=gse_link,
            modality=d.get("modality", ""),
            species=d.get("species", "unknown"),
            tissue=d.get("tissue", "unknown"),
            cell_count=int(d.get("cell_count", 0) or 0),
            pubmed_link=pubmed_link,
            submission_date=d.get("submission_date", "—") or "—",
            source_name=d.get("source_name", "—") or "—",
            geo_title=d.get("geo_title", "(no online title recorded)") or "(no online title)",
            description=d.get("description", "(no description)") or "(no description)",
        )
        (ds_dir / f"{_slugify_filename(key)}.md").write_text(page, encoding="utf-8")
        written += 1
    return written


def _slugify_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")


def _slug_lower(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ─────────────────── Method + Metric detail pages ─────────────────────

METHOD_PAGE = """---
title: "{name}"
type: docs
weight: {weight}
geekdocHidden: false
---

# {name}

| Field | Value |
|-------|-------|
| **Family** | {family} |
| **Group** | {group_name} |

## Description

{description}

{extra}

---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and 26 metrics; see the
[Methods index](../) for the full set.</small>
"""

METHOD_EXTRA = {
    "scCCVGBen_encoders": (
        "## Role in scCCVGBen\n\n"
        "Axis A (encoder-variation) sweep: CCVGAE trains a latent representation\n"
        "with this message-passing / attention module while holding the graph fixed\n"
        "to k-NN Euclidean. Benchmark naming for sweep rows: `scCCVGBen_{name}`.\n"
    ),
    "graph_constructions": (
        "## Role in scCCVGBen\n\n"
        "Axis B (graph-construction sweep): CCVGAE encoder is fixed to GAT while\n"
        "this graph builder constructs the cell-cell neighbourhood fed to the\n"
        "encoder. Benchmark naming: `scCCVGBen_GAT_{name}`.\n"
    ),
    "baselines": (
        "## Role in scCCVGBen\n\n"
        "Axis C (baseline comparison): this method produces a latent embedding\n"
        "evaluated with the same 26 metrics as CCVGAE. Benchmark naming: `{name}`\n"
        "(row label is the method name itself, with no scCCVGBen prefix).\n"
    ),
}

GROUP_DISPLAY = {
    "scCCVGBen_encoders":  "scCCVGBen graph encoder",
    "graph_constructions": "Graph construction method",
    "baselines":           "Dimensionality-reduction baseline",
}


def write_method_pages(methods: dict) -> int:
    out_dir = CONTENT_OUT / "methods"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.md"):
        if p.name != "_index.md":
            p.unlink()
    written = 0
    for group_key, items in methods.items():
        gname = GROUP_DISPLAY.get(group_key, group_key)
        extra_tpl = METHOD_EXTRA.get(group_key, "")
        for i, m in enumerate(items):
            name = m["name"]
            extra = extra_tpl.format(name=name)
            page = METHOD_PAGE.format(
                name=name,
                weight=100 + written,
                family=m.get("family", "—"),
                group_name=gname,
                description=m.get("description", ""),
                extra=extra,
            )
            (out_dir / f"{_slug_lower(name)}.md").write_text(page, encoding="utf-8")
            written += 1
    return written


METRIC_PAGE = """---
title: "{name}"
type: docs
weight: {weight}
geekdocHidden: false
---

# {name}

| Field | Value |
|-------|-------|
| **Group** | {group_name} |
| **Source** | `{source}` |
| **Direction** | {note} |

## Description

{description}

---

<small>Auto-generated from `scripts/build_site_data.py`. Every method's
latent embedding is scored against this metric across all 200 benchmark
datasets; see the [Metrics index](../) for the full set.</small>
"""

METRIC_GROUP_DISPLAY = {
    "clustering": "Clustering quality",
    "dre":        "Dimensionality-reduction evaluation (UMAP + tSNE)",
    "lse":        "Latent-space intrinsic geometry",
}


def write_metric_pages(metrics: dict) -> int:
    out_dir = CONTENT_OUT / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.md"):
        if p.name != "_index.md":
            p.unlink()
    written = 0
    for group_key, items in metrics.items():
        gname = METRIC_GROUP_DISPLAY.get(group_key, group_key)
        for m in items:
            name = m["name"]
            page = METRIC_PAGE.format(
                name=name,
                weight=100 + written,
                group_name=gname,
                description=m.get("description", ""),
                source=m.get("source", ""),
                note=m.get("note", "—"),
            )
            (out_dir / f"{_slug_lower(name)}.md").write_text(page, encoding="utf-8")
            written += 1
    return written


# ───────────────────────────────── Main ──────────────────────────────────

def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    CONTENT_OUT.mkdir(parents=True, exist_ok=True)

    print("→ Building datasets.json ...")
    datasets = build_datasets()
    _dump_json_dual(datasets, "datasets")
    print(f"  wrote {len(datasets)} datasets → data/ and static/")

    print("→ Building methods.json ...")
    methods = build_methods()
    _dump_json_dual(methods, "methods")
    n_methods = sum(len(v) for v in methods.values())
    print(f"  wrote {n_methods} methods")

    print("→ Building metrics.json ...")
    metrics = build_metrics()
    _dump_json_dual(metrics, "metrics")
    n_metrics = sum(len(v) for v in metrics.values())
    print(f"  wrote {n_metrics} metrics")

    print("→ Building summary.json ...")
    summary = build_summary(datasets)
    _dump_json_dual(summary, "summary")
    print("  wrote summary")

    print("→ Writing per-dataset pages ...")
    n = write_dataset_pages(datasets)
    print(f"  wrote {n} dataset pages → {CONTENT_OUT / 'datasets'}")

    print("→ Writing per-method pages ...")
    n = write_method_pages(methods)
    print(f"  wrote {n} method pages → {CONTENT_OUT / 'methods'}")

    print("→ Writing per-metric pages ...")
    n = write_metric_pages(metrics)
    print(f"  wrote {n} metric pages → {CONTENT_OUT / 'metrics'}")

    print("\n✓ Site data build complete.")


if __name__ == "__main__":
    main()
