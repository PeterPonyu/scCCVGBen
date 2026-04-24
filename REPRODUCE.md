# REPRODUCE.md — Reviewer Reproducibility Guide

From a clean clone to all figures. Intended for reviewers evaluating the scCCVGBen benchmark.

---

## Environment requirements

- Python 3.10 or later
- ~30 GB free disk space (h5ad symlinks + workspace caches + results)
- CUDA-capable GPU (tested: single NVIDIA GPU, ≥16 GB VRAM recommended)
- Internet access (STAGE 1 GEO/ENA downloads)
- Hugo ≥ 0.110 (for site build; optional for figure-only reproduction)

**Critical dependency**: `bash scripts/setup_symlinks.sh` requires
`/home/zeyufu/LAB/CCVGAE/` to be present on the same machine with its original directory
structure intact. The 55 scRNA + 100 scATAC h5ad files and pre-computed baseline results
live there and are symlinked rather than re-fetched. Without this directory the symlink
step will fail and you will need to re-run all 55+100 preprocessing steps manually
(not recommended; use the original host machine instead).

---

## Step-by-step

### 1. Clone and install

```bash
git clone <GITHUB_URL_PLACEHOLDER> scCCVGBen
cd scCCVGBen
pip install -e ".[dev]"
```

### 2. Verify imports

```bash
pytest tests/test_imports.py
```

### 3. STAGE 0 — Setup symlinks and build datasets.csv (~1 hr)

```bash
bash scripts/setup_symlinks.sh
python scripts/locate_scatac_baselines.py
python scripts/profile_existing_scrna.py
python scripts/build_datasets_csv.py --scan-host
```

### 4. STAGE 1 — Data acquisition (~12 hr, internet required)

```bash
python scripts/fetch_geo_scrna.py \
    --target 45 \
    --candidate-pool 60 \
    --out workspace/data/scrna/ \
    --candidate-csv data/scrna_candidate_pool.csv

python scripts/preprocess_scrna.py workspace/data/scrna/*_new.h5ad
python scripts/select_scatac_drops.py
python scripts/verify_benchmark_size.py
```

All 45 new scRNA datasets are fetched from public GEO/ENA accessions listed in
`data/scrna_candidate_pool.csv` (committed). Re-fetchable by anyone with internet access.

### 5. STAGE 2 — GPU sweep (~58 GPU-hours; can parallelize across panes)

```bash
python scripts/run_encoder_sweep.py
python scripts/run_graph_sweep.py
python scripts/run_baseline_backfill.py
```

All three are idempotent — re-run to resume after interruption.

### 6. STAGE 3 — Figures + site (~1 hr)

```bash
python scripts/make_overview_figure.py
python scripts/make_axisA_figure.py
python scripts/make_axisB_figure.py
python scripts/make_axisC_figure.py
python scripts/build_datasets_json.py
cd site && hugo --minify
```

---

## Expected outputs

| File | SHA-256 | Notes |
|---|---|---|
| `figures/fig_dataset_overview.pdf` | (populated after run) | dataset metadata overview |
| `figures/fig_axisA_encoder_ranking.pdf` | (populated after run) | Axis A encoder ranking |
| `figures/fig_axisB_graph_robustness.pdf` | (populated after run) | Axis B robustness |
| `figures/fig_axisC_baselines.pdf` | (populated after run) | Axis C baseline comparison |

SHA-256 hashes for committed figures will be added to this table after the full sweep
completes and figures are committed.

---

## Notes on reproducibility scope

- The 55 old scRNA + 100 scATAC h5ad files are **symlinked** from the prior CCVGAE work on
  the host machine. Their byte content is identical to what was used in the prior paper.
- The 55 old-scRNA × GAT baseline rows in Axis C and the GAT cells in Axis A for old scRNA
  are **reused** from `CG_results/CG_dl_merged/` (same host machine). These rows used the
  prior paper's hyperparameters; this asymmetry is documented in
  `results/encoder_sweep/README.md` and disclosed in the manuscript.
- All 45 new scRNA datasets are fetched fresh from public GEO/ENA; the fetch script
  (`scripts/fetch_geo_scrna.py`) records exact accession IDs and MD5s for each file.
- Random seeds are fixed in `training/configs/locked.yaml`; results should be numerically
  reproducible given the same hardware/driver versions.
