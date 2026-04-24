---
title: "About"
type: docs
weight: 90
---

# About scCCVGBen

scCCVGBen is a comprehensive benchmark for the CCVGAE family of methods on
200 single-cell omics datasets (100 scRNA-seq + 100 scATAC-seq), covering 14
graph encoder variants, 5 graph construction methods, and 13 dimensionality-
reduction baselines.

## Dataset curation

Every dataset in the benchmark has **GEO-verified online metadata** (species,
tissue, submission date, PubMed) fetched via GEOparse. The 100-scRNA
composition:

1. Prioritise tissues/conditions/organisms under-represented in existing
   CCVGAE work.
2. Prefer cellranger-standard filtered matrices (`filtered_feature_bc_matrix.h5`
   or per-GSM mtx trio `barcodes.tsv.gz + features.tsv.gz + matrix.mtx.gz`).
3. Small-and-clean preferred: 3k–30k cells (training subsamples to 3k).
4. Reproducibility floor: public GEO/ENA accession with raw counts.
5. Tiebreak: peer-reviewed publication preferred.

The 100-scATAC set: 115 canonical candidates, 15 dropped by lowest cell-count
strategy to match the 100-per-modality target.

Raw GEO metadata cache: `data/geo_metadata_cache.json`.

## Preprocessing (training-time)

| Modality | Pipeline |
|----------|----------|
| scRNA-seq | normalize_total(1e4) → log1p → 2,000 HVGs → subsample 3,000 cells |
| scATAC-seq | TF-IDF → top-2,000 HV peaks → LSI(50) → subsample 3,000 cells |

All preprocessing is implemented in `scccvgben/data/preprocessing.py` and
matches the CCVGAE revised-benchmark configuration bit-for-bit (see
`CCVGAE_supplement/run_hyperparam_sensitivity.py`).

## Citation

If you use scCCVGBen, please cite:

```bibtex
@article{scCCVGBen2026,
  title  = {scCCVGBen: a benchmark of graph variational autoencoders on
            single-cell RNA and ATAC data},
  author = {Fu, Zeyu and collaborators},
  year   = {2026},
  note   = {in preparation}
}
```

## Reproducibility

Source: [github.com/USER/scCCVGBen](https://github.com/USER/scCCVGBen)

```bash
git clone <repo>
cd scCCVGBen
pip install -e ".[dev]"
bash scripts/setup_symlinks.sh
python scripts/run_encoder_sweep.py --epochs 100   # Axis A (14 encoders)
python scripts/run_graph_sweep.py   --epochs 100   # Axis B (5 graphs)
python scripts/run_baseline_backfill.py            # Axis C (13 baselines)
python scripts/reconcile_result_schema.py          # merge into CCVGAE format
```

## License

MIT.
