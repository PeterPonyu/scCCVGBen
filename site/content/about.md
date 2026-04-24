---
title: "About"
type: docs
weight: 90
---

# About scCCVGBen

scCCVGBen is a benchmark on 200 single-cell omics datasets (100 scRNA-seq +
100 scATAC-seq). The site documents the datasets, methods, and metrics used
in the benchmark.

## Dataset metadata

Every dataset has GEO-verified metadata (species, tissue, submission date,
PubMed) fetched via GEOparse. Raw metadata cache:
`data/geo_metadata_cache.json`.

## Preprocessing (training-time)

| Modality | Pipeline |
|----------|----------|
| scRNA-seq | normalize_total(1e4) → log1p → 2,000 HVGs → subsample 3,000 cells |
| scATAC-seq | TF-IDF → top-2,000 HV peaks → LSI(50) → subsample 3,000 cells |

Source: `scccvgben/data/preprocessing.py`.

## Reproducibility

Source: [github.com/PeterPonyu/scCCVGBen](https://github.com/PeterPonyu/scCCVGBen).
See `REPRODUCE.md` in the repository for end-to-end instructions.

## License

MIT.
