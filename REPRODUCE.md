# Reproducing results

## Requirements

- Python 3.10+
- CUDA-capable GPU with ≥ 16 GB VRAM
- ~30 GB free disk (caches + intermediate h5ad files)
- Internet access for GEO/ENA fetches
- Hugo ≥ 0.124 for site build (optional)

## Steps

```bash
git clone https://github.com/PeterPonyu/scCCVGBen
cd scCCVGBen
pip install -e ".[dev]"
pytest tests/test_imports.py
```

### Data acquisition

```bash
python scripts/fetch_geo_scrna.py --out workspace/data/scrna/
python scripts/preprocess_scrna.py workspace/data/scrna/*_new.h5ad
```

### Sweeps

```bash
python scripts/run_encoder_sweep.py --epochs 100      # Axis A
python scripts/run_graph_sweep.py   --epochs 100      # Axis B
python scripts/run_baseline_backfill.py               # Axis C
```

All three are idempotent — re-run to resume.

### Site

```bash
python scripts/build_site_data.py
cd site && hugo --minify
```

Random seeds are pinned. Numerical reproducibility is expected on matching
hardware and driver versions.
