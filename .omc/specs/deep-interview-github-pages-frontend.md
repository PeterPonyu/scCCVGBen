# Deep Interview Spec: scCCVGBen GitHub Pages Frontend

## Metadata
- Interview rounds: 3
- Final ambiguity: 12.5% (threshold 20%)
- Type: brownfield (existing Hugo scaffold)
- Status: PASSED
- Generated: 2026-04-24 UTC

## Goal

Build a static **paper-companion + dataset-catalog** site for scCCVGBen,
covering the 200 benchmark datasets, 29 method variants, and 27 evaluation
metrics. Explicitly **exclude** computation-experiment results data
(benchmark run outputs) — the paper already covers those. The site
documents WHAT was benchmarked and WHY, not HOW methods performed.

## Constraints

- **Tech stack**: Hugo + `hugo-book` theme + Vega-Lite for charts
  (existing scaffold at `site/`, GH Actions `pages.yml` already deploys).
- **Deploy target**: GitHub Pages, `baseURL = "https://USER.github.io/scCCVGBen/"`.
- **No external-contribute pipeline** — site is read-only reference, not
  a live leaderboard.
- **No computation results** (method-metric tables) — paper-only, per
  user directive.

## Non-Goals

- ❌ Live leaderboard with user-sortable metric columns.
- ❌ Community method-contribute workflow (PapersWithCode-style).
- ❌ Replicating or hosting benchmark run outputs.
- ❌ Multi-language (English-only; zh content sources OK but render en).

## Information Architecture (4-chapter mode)

```
Home
├── Hero + abstract
├── Headline stats: 200 datasets, 29 methods, 27 metrics
├── Tissue breakdown pie chart (Vega-Lite)
├── Species × modality stacked bar (from benchmark_summary.json)
└── Citation + download

Datasets (200 items)
├── Top: filterable table (by modality, species, tissue)
│   — columns: filename_key, GSE, modality, species, tissue, cell_count, GEO link
│   — each row links to detail page
└── Detail pages (200): one per dataset
    — url: /datasets/{filename_key}/
    — content: all metadata + GEOparse title/submission_date/pubmed_id +
                description + local h5ad path + direct GEO URL

Methods (29 items in 3 groups)
├── scCCVGBen encoders (12): GAT, GATv2, Transformer, SuperGAT,
│                             GCN, SAGE, Graph, Cheb, TAG, ARMA, SG, SSG,
│                             GIN, EdgeConv
│   — per-method: short description + family (attention/message-passing) +
│                 citation + link to scCCVGBen source use
├── Graph constructions (4): kNN_euclidean, kNN_cosine, snn, mutual_knn,
│                             gaussian_threshold
│   — per-method: math definition + intuition
└── Baselines (13): PCA, KPCA, ICA, FA, NMF, TSVD, DICL, scVI, DIP,
                     INFO, TC, highBeta, CCVGAE
    — per-method: sklearn/scvi-tools class + citation

Metrics (27 items in 3 groups)
├── Clustering (6): ASW, DAV, CAL, NMI, ARI, COR
├── DRE UMAP/tSNE (10): distance_correlation, Q_local, Q_global, K_max,
│                        overall_quality — both UMAP and tSNE flavors
└── LSE intrinsic (10): manifold_dimensionality, spectral_decay_rate,
                         participation_ratio, anisotropy_score,
                         trajectory_directionality, noise_resilience,
                         core_quality, overall_quality, data_type,
                         interpretation
    — per-metric: formula + sklearn/DRE/LSE source reference

About
├── Paper abstract + full bibtex
├── Author list + affiliations
├── scCCVGBen repo link + license
└── Reproducibility: pip install + setup_symlinks invocation
```

## Acceptance Criteria

- [ ] Home page renders with Vega-Lite tissue pie + species/modality bars.
- [ ] Datasets listing page has filterable table (JS sort/filter via hugo-book
      or DataTables minimal JS); all 200 rows present.
- [ ] 200 per-dataset detail pages auto-generated from
      `data/benchmark_manifest.csv` using Hugo data-driven shortcode.
- [ ] Methods page groups 29 methods into 3 sub-sections (encoder, graph,
      baseline) with short description per method.
- [ ] Metrics page groups 27 metrics into 3 sub-sections with formula and
      source reference.
- [ ] About page has paper citation in BibTeX + repo link.
- [ ] `cd site && hugo --minify` produces a complete `public/` directory.
- [ ] GH Actions workflow deploys on push to main with no manual steps.
- [ ] hugo-book sidebar navigation auto-includes all sections.

## Technical Approach

1. **Data pipeline**: Python script `scripts/build_site_data.py` reads
   `data/benchmark_manifest.csv` + `data/benchmark_summary.json` + manually-
   curated `content/methods/*.md` and `content/metrics/*.md`, then:
   - Writes `site/data/datasets.json` with 200 dataset objects.
   - Writes `site/data/methods.json` with 29 method objects grouped by type.
   - Writes `site/data/metrics.json` with 27 metric objects grouped by type.
   - Writes/updates `site/content/datasets/*.md` one per dataset (auto-generated
     front-matter + rendered partial).

2. **Hugo layouts**:
   - `layouts/partials/dataset-card.html` (exists) — adapt for detail pages.
   - `layouts/shortcodes/dataset-table.html` (exists) — filterable table
     + Vega-Lite or alpine.js for filter.
   - `layouts/shortcodes/tissue-pie.html` (new) — Vega-Lite embed.
   - `layouts/shortcodes/species-modality-bar.html` (new).
   - `layouts/partials/method-card.html` (new).
   - `layouts/partials/metric-card.html` (new).

3. **Charts via Vega-Lite**:
   Embedded via `<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>`.
   Each shortcode emits a `<div>` + embedded spec JSON reading from
   `{{ site.Data.datasets }}`.

4. **Content authoring**:
   - Methods + metrics descriptions written in markdown with math via
     hugo-book's MathJax support.
   - Dataset pages generated, not hand-written.

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|---|---|---|---|
| Dataset | core | filename_key, GSE, modality, species, tissue, cell_count, description, pubmed_id, geo_url | many-to-many with Method via experiment runs |
| Method | core | name, family (encoder/graph/baseline), description, citation, source_library | — |
| Metric | core | name, formula, source (sklearn/DRE/LSE), group (cluster/DRE/LSE) | — |
| Page | supporting | route, layout, data_source | renders Dataset/Method/Metric |
| Chart | supporting | spec (Vega-Lite JSON), data_source | embedded in Home |

## Implementation Steps

1. Inspect + extend existing `site/` scaffold
2. Write `scripts/build_site_data.py` to generate JSON data files from
   benchmark_manifest.csv + summary.json
3. Generate `site/content/datasets/*.md` one per dataset
4. Author `site/content/methods/*.md` (3 sub-sections; 29 method cards)
5. Author `site/content/metrics/*.md` (3 sub-sections; 27 metric cards)
6. Add Vega-Lite chart shortcodes for Home
7. Test `hugo server` locally
8. Push → GH Actions deploys
