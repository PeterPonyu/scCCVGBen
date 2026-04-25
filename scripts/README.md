# scripts/

Each script is runnable with `python scripts/<name>.py --help`.
For the full execution sequence see `LAUNCH_BENCHMARK.md`.

## Setup & data curation

| Script | Purpose |
|---|---|
| `setup_symlinks.sh` | Idempotent: symlinks 55 scRNA baseline CSVs + 115 scATAC baseline CSVs from the external reference `CG_results` tree into `workspace/reused_results/`. Run once after clone. |
| `locate_scatac_baselines.py` | Verifies `CG_atacs/tables/` exists, lists CSVs, samples schema, writes `data/scatac_baseline_method_list.csv`. |
| `profile_existing_scrna.py` | Reads the 55 existing scRNA filenames, infers tissue/organism/category, writes `data/existing_scrna_diversity.csv` (used by `fetch_geo_scrna.py` to find gaps). |
| `fetch_geo_scrna.py` | **Skeleton** — CLI for downloading 45 new scRNA h5ad files from GEO/ENA. Real download logic is a TODO; dry-run mode prints candidate list. |
| `select_scatac_drops.py` | Picks 15 scATAC datasets to drop (lowest cell count + duplicate tissue), writes `data/dropped_scatac_v2.csv`. |
| `build_datasets_csv.py` | Two modes: `--scan-host` walks host dirs and writes source maps; `--build-canonical` builds `scccvgben/data/datasets.csv`. |
| `build_datasets_json.py` | Converts `datasets.csv` → `site/data/datasets.json` for Hugo. |

## Preprocessing

| Script | Purpose |
|---|---|
| `preprocess_scrna.py` | CLI wrapper for `scccvgben.data.preprocessing.preprocess_scrna`. Accepts one or more h5ad paths. |
| `preprocess_scatac.py` | Same for `preprocess_scatac`. |

## Verification

| Script | Purpose |
|---|---|
| `verify_benchmark_size.py` | Asserts `workspace/data/scrna/` has 100 h5ad and `workspace/data/scatac/` has 100 h5ad; exits non-zero with diagnostics on failure. |
| `make_figure1_site.py` | Builds `figures/fig1_scCCVGBen_site.png` and `.pdf` from local screenshots in `figures/site_shots/`. The screenshots are not committed. |

## Benchmark sweeps (GPU-bound; launch via LAUNCH_BENCHMARK.md)

| Script | Purpose |
|---|---|
| `run_encoder_sweep.py` | Axis A: 14 encoders × kNN-Euc(k=15) × 200 datasets. Skips GAT on 55 reused scRNA datasets. `--smoke` for quick test. |
| `run_graph_sweep.py` | Axis B: GAT × 5 graph constructions × 200 datasets. Skips kNN-Euc (shared with Axis A). `--smoke` flag. |
| `run_baseline_backfill.py` | Axis C: 13 baselines × 45 new scRNA only. Reused scRNA baselines come from symlinks. `--smoke` flag. |
