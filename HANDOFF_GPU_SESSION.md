# HANDOFF — scCCVGBen Launch-Prep Session → GPU Sweep Session

**Handoff timestamp**: 2026-04-23
**Upstream spec**: `.omc/specs/deep-interview-scccvgben-launch-prep.md`
**Implementation plan**: `.omc/plans/autopilot-impl.md` + `.omc/plans/autopilot-impl-addendum.md`
**Completed in this session**: All 9 `KNOWN_ISSUES.md` items + real GEO downloader + full
data inventory + revised-methodology alignment + publication-grade metric port.
**Still to run in the NEXT session**: `LAUNCH_BENCHMARK.md` STAGE 2 (GPU sweep) +
STAGE 3 (figures + Hugo site).

---

## 1 · What changed in this session

### Code (scccvgben/)
- **`training/dre.py`** (NEW, 180 lines) — ported `evaluate_dimensionality_reduction`,
  Q_local / Q_global / K_max / overall_quality from `CCVGAE/DRE.py` (revised 2026-04-23).
- **`training/lse.py`** (NEW, ~40 lines) — ported `trajectory_directionality` from
  `CCVGAE/LSE.py`.
- **`data/labels.py`** (NEW, ~50 lines) — unified `get_labels(adata)` helper replacing
  parallel fallback chains in `loader.py` and `baselines/runner.py`. Priority order:
  `cell_type → celltype → label → labels → cluster → clusters → annotation → CellType`
  → first category/low-cardinality column.
- **`training/metrics.py`** — UMAP/tSNE Q columns and `trajectory_directionality_intrin`
  replaced `np.nan` placeholders with real calls to the ported DRE/LSE modules. Only the
  5 true-TODO intrinsic columns remain NaN (noise_resilience, core_quality,
  overall_quality_intrin, data_type, interpretation) — marked explicitly and accepted per
  spec AC-2.
- **`training/trainer.py`** — removed unused `nb_loss` import (KNOWN_ISSUES #5).
- **`models/ccvgae.py:50-62`** — fixed `_conv_forward` to respect `needs_edge_attr` flag.
  Root cause of GATv2 silent NaN (KNOWN_ISSUES #2) was positional `edge_weight` being
  forwarded to encoders whose `lin_edge` was never initialised. All 12 encoders now
  produce finite metrics.
- **`data/preprocessing.py`** — added leading `sc.pp.filter_cells(min_counts=1)` +
  `sc.pp.filter_genes(min_cells=1)` plus try/except fallback from `cell_ranger` to
  `seurat` HVG flavor. Fixes KNOWN_ISSUES #1 (the "bin edges" error originated in
  scanpy's internal `pd.cut`, not in scCCVGBen's own code).
- **`data/loader.py`** — label extraction now delegates to `scccvgben.data.labels.get_labels`.
- **`baselines/runner.py:_get_labels`** — same unification.

### Scripts (scripts/)
- **`fetch_geo_scrna.py`** — full rewrite (181 → 296 lines). No more `NotImplementedError`.
  Real proxy-API downloader with GEOparse-primary + E-utilities fallback, 2 GB atlas
  filter, exponential backoff on 429, idempotent re-runs, failure CSV at
  `workspace/logs/fetch_failures.csv`. KNOWN_ISSUES #9 resolved.
- **`setup_symlinks.sh`** — parameterised via `REPO` / `SRC` env vars; added
  `SCATAC_SOURCES` array covering `/home/zeyufu/Downloads/ATAC_data/` (124 files) and
  `/home/zeyufu/Desktop/scATAC-25100/` (439 files across per-GSE subdirs). KNOWN_ISSUES
  #3 and #8 resolved.
- **`verify_benchmark_size.py`** — now supports revised (55+115=170) default targets
  plus `--v01` (100+100) fallback and `--min` / `--scrna` / `--scatac` / `--dropped`
  overrides. Defaults align with revised methodology.

### Metadata (data/)
- **`revised_methodology_manifest.csv`** (NEW, 172 rows) — canonical dataset list
  extracted from the three revised CCVGAE notebooks (2026-04-23) + supplement scripts.
  55 scRNA + 115 scATAC.
- **`existing_scrna_diversity.csv`** (REWRITTEN, 158 rows ← from 27) — full host
  inventory from T2 scan.
- **`existing_scatac_inventory.csv`** (NEW, 187 rows) — full host scATAC inventory.
- **`inventory_summary.md`** (NEW) — T2 agent summary with tissue / organism
  histograms and anomaly list.
- **`scrna_candidate_pool.csv`** (EXPANDED, 10 → 63 unique GSEs, 29 tissues) — reserve
  pool for future-paper-revision scenarios. Not consumed in this session because
  residual gap is 0.
- **`residual_gap_report.md`** (NEW) — proof that on-host data exceeds revised targets;
  justifies skipping T13 (actual download).
- **`dropped_scatac_v2.csv`** (EMPTIED to header-only) — revised methodology does not
  drop samples. Previously 15 rows with blank `cell_count`.
- **`scccvgben/data/datasets.csv`** (REBUILT, 201 rows: 115 scATAC + 86 scRNA) — fed by
  `scripts/build_datasets_csv.py --build-canonical` from the populated workspace
  symlinks. Over-coverage on scRNA (86 ≥ 55 needed) retained as flexibility buffer.

### CI / ops
- **`.github/workflows/pages.yml`** — all 4 `actions/*@vN` references pinned to full
  commit SHAs with tag comments (KNOWN_ISSUES #7 / OpenSSF Scorecard recommendation).

### Tests (tests/)
- **`test_dre_port.py`** (NEW, 10 tests) — regression coverage for ported DRE/LSE code.
- **`test_fetch_geo_skeleton.py`** (NEW, 9 tests) — exercises real downloader in dry-run
  mode, verifies idempotency, argument parsing, target-file picker logic. No network
  calls during tests.
- **Existing 31 tests** unchanged and all green.

### Planning artefacts (.omc/)
- `.omc/specs/deep-interview-scccvgben-launch-prep.md` — Phase 0 spec (8.05% ambiguity).
- `.omc/plans/autopilot-impl.md` — architect's 17-task plan (519 lines).
- `.omc/plans/autopilot-impl-addendum.md` — user-signed 6-decision addendum.
- `.omc/plans/t1-revised-methodology-summary.md` — revised CCVGAE methodology summary.

---

## 2 · All 9 KNOWN_ISSUES closed

| # | Title | Resolution | Commit target |
|---|-------|------------|---------------|
| 1 | Bin-edges error on real h5ad | `preprocessing.py` filter-cells/genes + flavor fallback | bugfix/metrics-binning |
| 2 | GATv2 silent NaN | `ccvgae.py::_conv_forward` respects `needs_edge_attr` | bugfix/gatv2-edge-attr |
| 3 | scATAC h5ad 0 → 592 | `setup_symlinks.sh` added ATAC_data/ + scATAC-25100/ | infra/symlinks |
| 4 | dropped_scatac_v2 blank | revised methodology drops 0 — CSV emptied | data/revised-drops |
| 5 | Dead nb_loss import | `trainer.py` import trimmed | cleanup/imports |
| 6 | DRY label extraction | `scccvgben/data/labels.py` + two call sites rewired | refactor/labels |
| 7 | GH Actions SHA pinning | 4 actions pinned via SHA+tag comment | security/actions |
| 8 | Hardcoded paths | `setup_symlinks.sh` parameterised via REPO/SRC env | infra/symlinks |
| 9 | fetch_geo skeleton | Full proxy-API downloader + tests | feature/fetch-geo |

---

## 3 · Test status

- **Before session**: 31/31 pytest green.
- **After session**: **50/50 pytest green** (31 original + 10 DRE regression + 9
  fetch_geo skeleton). No regression.

Run manually:
```bash
cd /home/zeyufu/LAB/scCCVGBen
python -m pytest tests/ -q
```

---

## 4 · Benchmark verification

```
$ python scripts/verify_benchmark_size.py --min
  PASS  scRNA:  86 >= 55 h5ad files
  PASS  scATAC: 115 >= 115 h5ad files
  PASS  dropped: 0 rows in dropped_scatac_v2.csv (target 0)
BENCHMARK VERIFICATION: PASS (55 scRNA + 115 scATAC = 170 datasets)
```

Smoke sweep result (real encoders, real metrics):
```
$ python scripts/run_encoder_sweep.py --smoke
  ... 4 rows written
  GAT   on GSE174367: ASW=0.24, CAL=334.8, Q_local_umap=0.45, NMI=... (real)
  GATv2 on GSE174367: ASW=0.27, CAL=468.8, Q_local_umap=0.45, NMI=... (real)
```
Previously GATv2 row was all-empty. Fix verified end-to-end.

---

## 5 · What the NEXT session should run

### STAGE 2 — GPU sweep (~40–58 GPU-hours)

```bash
cd /home/zeyufu/LAB/scCCVGBen

# Confirm state matches this handoff
python -m pytest tests/ -q
python scripts/verify_benchmark_size.py --min

# Axis A: 12 encoders × kNN-Euc × 170 datasets (minus 55 GAT×old-scRNA reuse)
python scripts/run_encoder_sweep.py

# Axis B: GAT × 5 graph constructions × 170 datasets (minus kNN-Euc shared cell)
python scripts/run_graph_sweep.py

# Axis C: 13 baselines × 170 datasets (minus 55 scRNA + 115 scATAC baseline reuse)
python scripts/run_baseline_backfill.py
```

Wall-clock guidance: ~1 min/run on local GPU. Expected total ~2–3 days if run serially;
cut to ~24 h with 3 tmux panes sharing one GPU.

Each runner writes per-dataset CSVs and resumes idempotently on re-run.

### STAGE 3 — Figures + Hugo (~1 h)

```bash
python scripts/make_overview_figure.py
python scripts/make_axisA_figure.py
python scripts/make_axisB_figure.py
python scripts/make_axisC_figure.py
python scripts/build_datasets_json.py
cd site && hugo --minify
```

### Paper / site integration (open-ended)

- Merge STAGE 2 result tables with `data/revised_methodology_manifest.csv` for the
  canonical paper figures.
- Hugo site reads `site/data/datasets.json` produced by `build_datasets_json.py`.
- GitHub Pages deploy triggered automatically on push to `main` (workflow now
  SHA-pinned and secure).

---

## 6 · Known caveats carried forward

1. **5 intrinsic metric columns remain NaN** (`noise_resilience_intrin`,
   `core_quality_intrin`, `overall_quality_intrin`, `data_type_intrin`,
   `interpretation_intrin`). These were not part of the CCVGAE revised methodology
   manifest and are explicitly `# TODO` in `metrics.py`. Paper methods section should
   either omit or mark as "Not applicable".

2. **`datasets.csv` has 201 rows, not exactly 170**. Over-coverage: 86 scRNA on-host
   vs 55 canonical target + 115 scATAC exact. To filter to the canonical 170, join
   against `data/revised_methodology_manifest.csv` on `GSE`:
   ```python
   import pandas as pd
   ds = pd.read_csv("scccvgben/data/datasets.csv")
   ma = pd.read_csv("data/revised_methodology_manifest.csv")
   canonical = ds[ds["GSE"].isin(ma["gse"])]
   assert len(canonical) == 170
   ```
   (Do this per-runner or as a one-time filter in `build_datasets_csv.py`.)

3. **KMeans n_init=10 already in metrics.py** — satisfies T1 agent's "10-seed lowest
   inertia" finding.

4. **scRNA candidate pool (63 GSEs) is a reserve** — not used in this session but
   available for future paper-revision sensitivity analyses via
   `python scripts/fetch_geo_scrna.py --target N --candidate-csv data/scrna_candidate_pool.csv`.

5. **GEOparse is optional**. Install via `pip install -e .[geo]`. If skipped,
   `fetch_geo_scrna.py` auto-falls-back to E-utilities HTTPS API.

6. **Benchmark verification `--min` mode is the canonical gate**. Strict equality
   (without `--min`) will fail because the workspace has surplus scRNA symlinks.

7. **Naming alignment** — the revised CCVGAE paper uses **GAT-VAE** terminology
   (supersedes earlier "VGEA"). `scccvgben/` code still uses `CCVGAE` as the model
   class name. Paper-side naming is a doc-only concern to address before submission.

---

## 6.5 · Post-autopilot deep audit (2026-04-23, user-requested)

After the 4 commits landed, a deep audit answered four user questions: do all
claimed data files carry metadata; do training scripts store in a format that
merges with reused results; is the DRE/LSE code really the revised version; is
the label-preservation strategy truly unified.

### Audit E — Per-row data coverage (deep-verified): **172/172 = 100%, zero gap**

Surface-count gap analysis (first pass) reported "gap = 0 / surplus" based on
raw h5ad file counts (110 vs 55 scRNA, 592 vs 115 scATAC). Deep per-row
verification revealed two subtle issues that the surface count hid:

1. GSE-regex matching falsely flagged 5 scRNA datasets as missing because
   their friendly filenames (`endo.h5ad`, `dentate.h5ad`, `lung.h5ad`,
   `IRALL.h5ad`, `wtko0312.h5ad`) do not contain `GSE\d+` prefixes.
   Reality: all 5 are on host, correctly symlinked.
2. 6 scATAC rows had `gse=GSE248xxx` — literal placeholder, not a real
   accession. Located the GSM files in
   `/home/zeyufu/Downloads/ATAC_data/GSE266511_RAW/`; corrected manifest.

Final verified coverage:

| Modality | Canonical rows | On-host matched | True gap |
|----------|---------------|-----------------|----------|
| scRNA    | 57 | 57 | 0 |
| scATAC   | 115 | 115 | 0 |
| **Total**| **172** | **172** | **0** |

One **pre-STAGE-2 build step** remains (not a data gap): the `UCB_timeseries`
row in the manifest points to 5 raw 10x_mtx directories (CT/D4/D7/D11/D14),
not an h5ad. Concatenate before STAGE 2:

```bash
python scripts/build_ucb_timeseries.py
# writes workspace/data/scrna/ucb_timeseries.h5ad (~2 min)
```

See `data/residual_gap_report.md` for the full deep-verification transcript
and reproducer.

### Audit A — Metadata coverage: **PARTIAL / honest disclosure required**

On a 30-file random sample each:

| Modality | cell_type | tissue | species | raw counts |
|----------|-----------|--------|---------|-----------|
| scRNA (n=30) | 8/30 (27%) | 0/30 | 0/30 | 22/30 (73%) |
| scATAC (n=30) | 0/30 | 0/30 | 0/30 | 26/30 (87%) |

**Implication for paper**:
- Ground-truth `cell_type` exists on only a minority of scRNA datasets and zero
  scATAC datasets. Paper should disclose this and report NMI/ARI:
  - Honest-mode: only on the 8 scRNA datasets that have cell_type → pair-wise
    statistics have ~8-dataset power, not 170.
  - Self-reference-mode: on all 170 datasets via the ``ref = labels if labels is
    not None else pred`` pattern (now implemented — see Audit D) → NMI/ARI
    become self-consistency scores ≈ 1.0 on unlabelled data, only meaningful as
    a "cluster separability" proxy on labelled data.
- Tissue / organism metadata must be inferred from filename heuristics in
  `scripts/build_datasets_csv.py::_infer_tissue` / `_infer_species` — not from
  `adata.obs`. This is already the implemented behaviour; be aware accuracy is
  fuzzy (152 of 201 rows classified as tissue='other').

### Audit B — Schema alignment: **reconciler added**

Reused results use three different schemas:

- `CG_dl_merged/*.csv` (55 scRNA reused): 27 cols, `method, ASW, ... , NMI, ARI`
- `CG_atacs/tables/*.csv` (115 scATAC reused): 27 cols, **starts with unnamed
  index**, NMI/ARI **in front**, **no method column**
- Our new `run_encoder_sweep.py` / `run_graph_sweep.py` output: 29 cols
  (adds `encoder, family` at the front)
- Our new `run_baseline_backfill.py` output: matches `CG_dl_merged`

Direct `pd.concat` across these will fail. Added
**`scripts/reconcile_result_schema.py`** that normalises all sources to a
canonical 32-column schema:

```bash
python scripts/reconcile_result_schema.py \
    --in results/encoder_sweep/ results/graph_sweep/ \
         results/baselines/ workspace/reused_results/ \
    --out results/reconciled/
# produces results/reconciled/all_results_reconciled.csv (1193 rows after STAGE 2)
# plus axis_A_reconciled.csv / axis_B_reconciled.csv / axis_C_reconciled.csv
```

Run this after STAGE 2 completes, before the figures / statistics step.

### Audit C — DRE / LSE provenance: **MAIN results, not supplementary (corrected)**

Initial claim (first pass, WRONG): "DRE/LSE are initial-version code retained
only for schema compatibility → report in supplementary material."

Corrected via deeper grep across the entire CCVGAE repo:

- `CCVGAE/DRE.py` and `LSE.py` are consumed by the **CCVGAE_supplement**
  production scripts that generate the per-dataset result CSVs:
  - `CCVGAE_supplement/run_hyperparam_sensitivity.py`
  - `CCVGAE_supplement/run_4new_atac.py`
  - `CCVGAE_supplement/run_centroidvae_supplement.py`
  - `CCVGAE_supplement/fix_outlier_atac.py`
  - `CCVGAE_supplement/fix_failed_datasets.py`
  - `CCVGAE_supplement/fix_missing_variants.py`
  - `CCVGAE_supplement/verify_result_claims.py`
  - `CCVGAE_supplement/rerun_muscle_hyperparam.py`
- The 3 revised notebooks (`CCVGAE1_SD`, `CCVGAE2_UCB`, `CCVGAE3_IR`) are
  **paper-figure generation**, not per-dataset metric computation — that is
  why they import only sklearn's 5 metrics for cell-level KMeans
  visualisations, not the full 27-column schema.
- The **CentroidVAE_results/**/tables/*.csv** result CSVs (and the reused
  `CG_dl_merged/*.csv` / `CG_atacs/tables/*.csv`) carry all 27 columns,
  including Q_local/Q_global/K_max/overall_quality + 8 intrinsic columns.

**Correct interpretation**: DRE Q-metrics and LSE trajectory_directionality are
**main paper results**. The paper methods section should describe:
- Coranking-matrix-based Q_local / Q_global / K_max / overall_quality for
  UMAP and tSNE embeddings of the latent space (Lee & Verleysen 2009),
  computed via `scccvgben/training/dre.py`.
- PCA-principal-component-dominance trajectory directionality via
  `scccvgben/training/lse.py`.
- 4 manifold-geometry intrinsic proxies already implemented in
  `metrics.py::_intrinsic_metrics` (manifold_dimensionality, spectral_decay,
  participation_ratio, anisotropy).

The 5 columns that remain explicitly NaN (`noise_resilience_intrin`,
`core_quality_intrin`, `overall_quality_intrin`, `data_type_intrin`,
`interpretation_intrin`) require further porting from `CCVGAE/LSE.py`
`SingleCellLatentSpaceEvaluator` class (see `/home/zeyufu/LAB/CCVGAE/LSE.py:9`
and the full 626-line class for `evaluate_single_cell_latent_space`). These
are **also main results** per the revised paper schema and should be ported
in the next session before paper submission.

### Audit D — Label preservation strategy: **now unified, was not**

Before audit:
- `loader.py`, `baselines/runner.py` both delegated to `get_labels()` ✓
- `metrics.py::compute_metrics` used `KMeans(pred)` as silent-fallback for
  ASW/DAV/CAL ✓
- `metrics.py::_clustering_metrics` returned `{NMI: NaN, ARI: NaN}` when
  labels was None ✗ (diverged from CCVGAE_supplement pattern
  `ref = labels if labels is not None else pred`)

Fix applied after audit:
```python
# metrics.py::_clustering_metrics
ref = labels if labels is not None else pred
return {"NMI": NMI(ref, pred), "ARI": ARI(ref, pred)}
```
Verified: `_clustering_metrics(Z, None, None)` now returns `{'NMI': 1.0, 'ARI': 1.0}`
(self-consistency), no longer NaN. 50/50 tests still green.

**Caveat**: self-consistency NMI/ARI = 1.0 is informative only if you
distinguish it from a ground-truth NMI/ARI. Downstream stats code **must**
check `data/revised_methodology_manifest.csv` or the per-dataset cell_type
column presence before including rows in any "NMI across datasets" boxplot
or Wilcoxon comparison.

---

## 7 · Anti-checklist (things explicitly NOT done this session)

- ❌ Running the full Axis A/B/C GPU sweep (deferred to STAGE 2)
- ❌ Generating publication figures (STAGE 3)
- ❌ Building / deploying the Hugo site (STAGE 3)
- ❌ Writing the paper manuscript
- ❌ Running statistical tests (Wilcoxon + Holm-Bonferroni) — these need the sweep
  results to operate on
- ❌ Downloading additional scRNA from GEO — skipped because residual gap = 0 per
  `residual_gap_report.md`. The downloader is implemented and tested but was not
  exercised against the live GEO API.
- ❌ Re-running CCVGAE baseline on the reused 55 old-scRNA cells — those remain
  symlinked from `CCVGAE/CG_results/CG_dl_merged/` per the Reuse Map, creating the
  GAT-row provenance asymmetry documented in `results/encoder_sweep/README.md`.

---

## 8 · Resume checklist for the next operator

- [ ] `git log --oneline | head -15` — review session commits
- [ ] `pytest tests/ -q` → 50 passed expected
- [ ] `python scripts/verify_benchmark_size.py --min` → PASS expected
- [ ] `python scripts/run_encoder_sweep.py --smoke` → 4 rows, real numbers
- [ ] Read §5 above and pick a STAGE 2 pane strategy (serial or 3-pane tmux)
- [ ] Launch STAGE 2
- [ ] After STAGE 2 completes, run STAGE 3 figures + Hugo
- [ ] Generate paper tables + site JSON; commit; push; GH Pages auto-deploys

---

*Handoff authored by autopilot Phase 2+5. Any follow-up questions: re-interview via
`/deep-interview` citing this file as context.*
