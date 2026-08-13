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

## Cite

**scCCVGBen for Benchmarking of Single-Cell Representation Learning Anchored on a Centroid-Coupled Variational Graph Attention Autoencoder across scRNA-seq and scATAC-seq**

Fu, Z.<sup>#</sup>, Fu, J.<sup>#</sup>, Chen, C.<sup>#</sup>, Zhang, K., Wang, J.<sup>*</sup>, Ran, T.<sup>*</sup>, Wang, S.<sup>*</sup>

*Frontiers in Genetics* · 2026

- [DOI](https://doi.org/10.3389/fgene.2026.1822168)
- [Code](https://github.com/PeterPonyu/scCCVGBen)
- [Site](https://peterponyu.github.io/scccvgben-next/)

## Reproducibility

Source: [github.com/PeterPonyu/scCCVGBen](https://github.com/PeterPonyu/scCCVGBen).
See `REPRODUCE.md` in the repository for end-to-end instructions.

## License

MIT.
