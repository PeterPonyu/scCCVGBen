# Known Issues (Phase 5 follow-ups)

These are non-blocking issues surfaced during autopilot validation. None prevent the user from launching the benchmark via `LAUNCH_BENCHMARK.md`, but they should be fixed before public release or paper submission.

## High priority (fix before STAGE 2 GPU sweep)

### 1. "Bin edges must be unique" error on real h5ad datasets
- **Symptom**: `scripts/run_encoder_sweep.py` and `run_graph_sweep.py` complete with rc=0 on real datasets but write rows with empty metric columns. Error logged: `Bin edges must be unique`.
- **Likely cause**: pandas `cut`/`qcut` call somewhere in the metrics path encounters duplicate bin edges on small/sparse latents.
- **Fix**: locate the binning call site (probably in `scccvgben/training/metrics.py` Q-metric or trajectory_directionality computation) and pass `duplicates="drop"`.

### 2. GATv2 encoder still fails at runtime (caught silently)
- **Symptom**: `results/encoder_sweep/synthetic_200x100.csv` shows the GATv2 row with all-NaN metrics. Other 12 encoders produce real numbers.
- **Likely cause**: GATv2Conv requires specific kwargs that build_encoder may not be passing correctly.
- **Fix**: inspect `scccvgben/models/encoder_registry.py` GATv2 entry; verify required init args (heads, dropout, etc.).

## Medium priority (cleanup)

### 3. 0 scATAC h5ads located on this host
- **Symptom**: `setup_symlinks.sh` reports h5ad scATAC: 0.
- **Effect**: Axis A (encoder sweep) and Axis B (graph sweep) cannot run on the 100 scATAC half of the benchmark from STAGE 2 onward.
- **Action**: User locates the scATAC h5ads (consult `archived_extra_scATAC/` filenames in `/home/zeyufu/LAB/CCVGAE/CG_results/` to know what filenames to search for) and adds their source dir to `SCRNA_SOURCES` in `setup_symlinks.sh:84+`.

### 4. dropped_scatac_v2.csv has blank cell_count
- **Symptom**: 15 drop rows have empty cell_count + tissue=other.
- **Cause**: Cell counts read from h5ad files; with 0 scATAC h5ads on host, can't fill.
- **Fix**: After issue #3 is resolved, re-run `python scripts/select_scatac_drops.py` to populate.

## Low priority (polish)

### 5. Dead import in trainer.py
- `from .losses import nb_loss` is unused after the MSE switch.

### 6. DRY violation: label-extraction logic duplicated
- `scccvgben/data/loader.py:60-77` and `scccvgben/baselines/runner.py:_get_labels` both walk the same fallback chain (`cell_type` → `leiden` → `celltype` → ...). Extract a helper.

### 7. GitHub Actions not pinned to SHAs
- `.github/workflows/pages.yml` uses `actions/checkout@v4` etc. Pin to SHAs before public release per OpenSSF Scorecard recommendations.

### 8. setup_symlinks.sh hardcodes /home/zeyufu paths
- Parameterize via `REPO="${REPO:-$(git rev-parse --show-toplevel)}"`, `SRC="${SRC:?...}"` for portability.

### 9. fetch_geo_scrna.py is a skeleton
- Real GEO/ENA download not implemented. Either wire up `GEOparse.get_GEO` or document that the user must implement before STAGE 1.
