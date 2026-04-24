# LAUNCH_BENCHMARK.md

Run this after autopilot Phase 2 completes (all code files written). These stages take the
repo from scaffold to full results + site.

---

## Prerequisites

- Python 3.10+ with `pip install -e ".[dev]"` already done
- `/home/zeyufu/LAB/CCVGAE/` present on the same machine (symlink source)
- Internet access for STAGE 1 (GEO downloads)
- GPU with CUDA for STAGE 2

---

## STAGE 0 — Setup (~1 hour, no GPU needed)

Establish symlinks, locate existing result files, profile the 55 existing scRNA datasets for
diversity-gap analysis, and build an initial datasets.csv from on-disk h5ad files.

```bash
# 1. Establish all Reuse Map symlinks (idempotent; safe to re-run)
bash scripts/setup_symlinks.sh

# 2. Locate scATAC baseline tables and verify schema match
python scripts/locate_scatac_baselines.py

# 3. Profile existing 55 scRNA (tissue/condition/organism histogram)
python scripts/profile_existing_scrna.py

# 4. Build datasets.csv from h5ad files found on disk
python scripts/build_datasets_csv.py --scan-host
```

Expected output: `scccvgben/data/datasets.csv` populated with 55 scRNA + 100 scATAC entries
(the 45 new scRNA rows are added after STAGE 1). Symlinks under `workspace/` verified.

---

## STAGE 1 — Data Acquisition (~12 hours, internet-bound)

Fetch 45 new scRNA datasets from GEO/ENA, preprocess them, finalize the scATAC drop list,
and assert the 200-dataset benchmark size.

```bash
# 1. Fetch 45 new scRNA from GEO/ENA (diversity-first, per inclusion rubric in spec)
python scripts/fetch_geo_scrna.py \
    --target 45 \
    --candidate-pool 60 \
    --out workspace/data/scrna/ \
    --candidate-csv data/scrna_candidate_pool.csv

# 2. Preprocess newly fetched h5ad files
#    (libnorm → log1p → 2000 HVGs → PCA(50))
python scripts/preprocess_scrna.py workspace/data/scrna/*_new.h5ad

# 3. Decide 15 scATAC drops (lowest cell count + duplicate tissue criterion)
python scripts/select_scatac_drops.py

# 4. Assert exactly 100 scRNA + 100 scATAC = 200
python scripts/verify_benchmark_size.py
```

Expected wall-clock: GEO downloads dominate (~10-11 hr depending on bandwidth and
dataset sizes); preprocessing ~30-60 min on CPU.

**Resuming after interruption**: `fetch_geo_scrna.py` is idempotent — re-running skips
already-downloaded files. `preprocess_scrna.py` skips files whose output h5ad already exists.

---

## STAGE 2 — GPU Sweep (~58 GPU-hours serial; split across tmux panes to parallelize)

Run all three axes. These can be launched concurrently in separate terminal panes if GPU
memory allows (each encoder/graph sweep uses one GPU; baseline backfill is CPU-heavy but
small).

```bash
# Pane 1 — Axis A: 12 encoders × kNN-Euc × 200 datasets
#   Skips the 55 historical GAT × old-scRNA cells (reused from CG_dl_merged via symlink)
python scripts/run_encoder_sweep.py

# Pane 2 — Axis B: GAT × 5 graph constructions × 200 datasets
#   kNN-Euc cell shared with Axis A; 4 new graph constructions × 200 = 800 new runs
python scripts/run_graph_sweep.py

# Pane 3 — Axis C: 13 baselines × 45 new scRNA only (585 new runs)
#   scATAC baselines are all reused (0 new runs); old scRNA baselines reused from symlinks
python scripts/run_baseline_backfill.py
```

Expected wall-clock (serial, single GPU):
- Axis A: ~2,345 runs × 1 min ≈ 39 GPU-hours
- Axis B: ~800 runs × 1 min ≈ 13 GPU-hours
- Axis C: ~585 runs × 1 min ≈ 10 GPU-hours (mostly CPU; scVI/deep methods use GPU)

**Total serial**: ~62 GPU-hours (~2.5 days). With 3 concurrent panes on a single GPU
(encoder sweep first, then graph + baseline in parallel once encoder finishes the shared
kNN-Euc cells): ~40-45 hours wall-clock.

**Resuming after interruption**: all three sweep scripts write results per-dataset and skip
datasets whose output CSV already exists. Re-run the same command to pick up where it left off.

### GAT-row provenance asymmetry (important for paper methods section)

The 55 old-scRNA × GAT cells in Axis A are **reused from prior CCVGAE work** (symlinked
from `CG_results/CG_dl_merged/`). These cells used the historical hyperparameters from the
prior paper. The 145 other encoder cells (including GAT on 45 new scRNA) use the
hyperparameters from `training/configs/locked.yaml`. This asymmetry is:

- Documented in `results/encoder_sweep/README.md`
- Accepted by the user (Round 7 of the spec interview) for efficiency
- Must be disclosed in any paper/preprint generated from these results

---

## STAGE 3 — Figures + Site (~1 hour)

Generate all publication figures and build the Hugo static site.

```bash
# Generate figures (each script is independent; run in any order)
python scripts/make_overview_figure.py
python scripts/make_axisA_figure.py
python scripts/make_axisB_figure.py
python scripts/make_axisC_figure.py

# Generate site data JSON from datasets.csv
python scripts/build_datasets_json.py

# Build Hugo site
cd site && hugo --minify
```

Expected output:
- `figures/fig_dataset_overview.{pdf,png}`
- `figures/fig_axisA_encoder_ranking.{pdf,png}`
- `figures/fig_axisB_graph_robustness.{pdf,png}`
- `figures/fig_axisC_baselines.{pdf,png}`
- `site/public/` (static site, gitignored; deployed via GitHub Actions on push to main)

---

## Summary of expected outputs

| Artifact | Location | Notes |
|---|---|---|
| datasets.csv | `scccvgben/data/datasets.csv` | 200-row source of truth |
| Axis A results | `results/encoder_sweep/{dataset_key}.csv` | 200 files |
| Axis B results | `results/graph_sweep/{dataset_key}.csv` | 200 files |
| Axis C scRNA | `results/baselines/scrna_{dataset_key}.csv` | 100 files |
| Axis C scATAC | `results/baselines/scatac_{dataset_key}.csv` | 100 files |
| Stats outputs | `results/stats/*.csv` | 3 files |
| Figures | `figures/*.{pdf,png}` | 8 files |
| Static site | `site/public/` | gitignored; CI deploys |
