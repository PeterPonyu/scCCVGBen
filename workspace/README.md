# workspace/ — input data + reused results + runtime artefacts

**Everything under `workspace/` is gitignored** (large binaries, symlinks, logs).
Only the directory structure and `.gitkeep` files are committed.

Run `bash scripts/setup_symlinks.sh` from the repo root to (re)populate this
directory; the script is idempotent.

---

## Layout

```
workspace/
├── data/                           — Input data layer (training reads from here)
│   ├── scrna/                      ▶ unified scRNA view: symlinks from BOTH
│   │                                 (a) on-host raw files (via SCRNA_SOURCES
│   │                                     in setup_symlinks.sh) AND
│   │                                 (b) freshly downloaded files (from
│   │                                     scrna_geo/ below)
│   ├── scatac/                     ▶ unified scATAC view: same pattern, from
│   │                                 SCATAC_SOURCES + scatac_geo/
│   ├── scrna_geo/                  ▶ real download destination for
│   │                                 fetch_geo_scrna.py — files named
│   │                                 {gse}_new.h5ad. Kept separate from
│   │                                 scrna/ to preserve provenance at a glance.
│   └── scatac_geo/                 ▶ symmetric staging dir; unused in v0.1
│                                     (scATAC downloader not implemented).
│
├── reused_results/                 — Symlinks into prior CCVGAE run tables
│   │                                 (these are the 225 reused cells that
│   │                                 avoid re-running Axis A/C on known cells)
│   ├── scrna_baselines/            ▶ 55 CSVs — scRNA baseline reference;
│   │                                 source = /home/zeyufu/LAB/CCVGAE/
│   │                                         CG_results/CG_dl_merged/
│   ├── axisA_GAT_scrna/            ▶ 55 CSVs — Axis-A GAT row on old scRNA;
│   │                                 same source as above, GAT row extracted
│   │                                 at consumption time by run_encoder_sweep.py
│   └── scatac_baselines/           ▶ 115 CSVs — scATAC baseline reference;
│                                     source = CG_results/CG_atacs/tables/
│
├── checkpoints/                    — Model .pt checkpoints from training
│                                     (v0.1 does NOT persist ckpts; this is a
│                                      placeholder for future paper revisions)
│
├── geo_cache/                      — GEOparse metadata cache (.soft files);
│                                     re-use across fetch_geo_scrna.py runs
│                                     to avoid repeated NCBI hits.
│
└── logs/                           — Per-script logs; persistent across runs.
    ├── fetch_geo.log               ▶ downloader events, successes, failures
    └── fetch_failures.csv          ▶ machine-readable failure list for
                                       retrying failed GSEs next session
```

---

## Naming conventions

| Pattern | Where | Meaning |
|---------|-------|---------|
| `{gse}_new.h5ad`         | `data/scrna_geo/`  | fresh GEO download |
| `{gse}_raw_{filename}`   | `data/scrna_geo/`  | raw 10x h5 before scanpy conversion (kept for traceability) |
| `{friendly_name}.h5ad`   | `data/scrna/`      | symlink to on-host raw data (e.g., `endo.h5ad`, `IRALL.h5ad`) |
| `GSE\d+__*.h5`           | `data/scatac/`     | on-host scATAC symlink from scATAC-25100/{N}-{GSE}/ |
| `{Can|Dev|...}_{gse}_{desc}_df.csv` | `reused_results/scrna_baselines/` | prior-run 27-column metric CSV |

---

## Cardinality after `bash scripts/setup_symlinks.sh`

| Sub-path | Expected count | Source |
|----------|---------------|--------|
| `data/scrna/`              | 86 + N_geo   | on-host SCRNA_SOURCES (86) + fresh downloads (N_geo) |
| `data/scatac/`             | 592          | on-host SCATAC_SOURCES |
| `reused_results/scrna_baselines/`  | 55  | CCVGAE CG_dl_merged |
| `reused_results/axisA_GAT_scrna/`  | 55  | same source, GAT row |
| `reused_results/scatac_baselines/` | 115 | CCVGAE CG_atacs/tables |

To reach the paper-grade 100-scRNA target, the GPU session should first run:

```bash
python scripts/fetch_geo_scrna.py --target 22 \
    --candidate-csv data/scrna_candidate_pool.csv \
    --out workspace/data/scrna_geo/
bash scripts/setup_symlinks.sh   # re-link scrna_geo files into scrna/
```

(Target of 22 comes from `data/residual_gap_report.md` deduplication analysis:
86 surface files → 78 unique → gap = 22 vs 100 target. Candidate pool has
63 unique GSEs providing ~2.9× buffer.)

---

## Script I/O contract summary

Scripts read from `workspace/data/*` and `workspace/reused_results/*`; they
write results to `results/*` (see `results/README.md`).

| Script | Reads from | Writes to |
|--------|-----------|-----------|
| `fetch_geo_scrna.py`       | `data/scrna_candidate_pool.csv`        | `workspace/data/scrna_geo/` + `workspace/logs/` |
| `build_ucb_timeseries.py`  | `/home/zeyufu/Desktop/UCBfiles/`       | `workspace/data/scrna/ucb_timeseries.h5ad` |
| `build_datasets_csv.py`    | `workspace/data/*/`                    | `scccvgben/data/datasets.csv` |
| `run_encoder_sweep.py`     | `workspace/data/scrna/ + scatac/`, `reused_results/axisA_GAT_scrna/` | `results/encoder_sweep/{key}.csv` |
| `run_graph_sweep.py`       | `workspace/data/scrna/ + scatac/`      | `results/graph_sweep/{key}.csv` |
| `run_baseline_backfill.py` | `workspace/data/scrna/`, `reused_results/scrna_baselines/` | `results/baselines/{modality}_{key}.csv` |
| `reconcile_result_schema.py` | `results/*/ + workspace/reused_results/*/` | `results/reconciled/` |
