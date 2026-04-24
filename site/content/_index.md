---
title: "scCCVGBen"
type: docs
---

# scCCVGBen — Single-cell CCVGAE Benchmark

**scCCVGBen** is a curated benchmark suite of **200 single-cell omics datasets**
(100 scRNA-seq + 100 scATAC-seq) designed to evaluate the CCVGAE family of
methods along three axes: **14 graph encoder variants**, **5 graph construction
choices**, and **13 dimensionality-reduction baselines**.

This site documents the *benchmark itself* — the datasets, the compared
methods, and the evaluation metrics. For the numerical results and the
statistical analysis, please refer to the accompanying paper.

---

## Benchmark at a glance

{{< headline-stats >}}

---

## Benchmark composition

{{< home-composition >}}

---

## Metadata distributions

{{< extended-stats >}}

---

## Explore

- [**Datasets →**]({{< ref "/datasets" >}}) Browse all 200 benchmark datasets
  with per-dataset detail pages (GEO-verified metadata, PubMed, GEO submission
  date).
- [**Methods →**]({{< ref "/methods" >}}) 14 scCCVGBen graph encoders
  (attention + message-passing), 5 graph construction methods, and 13 baseline
  dimensionality-reduction algorithms.
- [**Metrics →**]({{< ref "/metrics" >}}) 26 evaluation metrics: 6 clustering
  quality, 10 coranking-based DRE, 10 latent-space intrinsic-geometry.
- [**About →**]({{< ref "/about" >}}) Paper citation, reproducibility pointers,
  license.
