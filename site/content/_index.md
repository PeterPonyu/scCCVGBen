---
title: "scCCVGBen"
type: docs
---

# scCCVGBen — Single-cell scCCVGBen Benchmark

**scCCVGBen** is a curated benchmark suite of **200 single-cell omics datasets**
(100 scRNA-seq + 100 scATAC-seq) designed to evaluate the scCCVGBen family of
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

## Model architecture

Figure 2 summarizes the model-first path through scCCVGBen: input preprocessing,
five graph-construction choices, the 14-encoder registry, variational latent
flow, dual reconstruction outputs, and the grouped 20-display-metric evaluation
table.
The publication image is rendered locally by
`scripts/make_figure2_model_architecture.py`; generated figure binaries are not
checked into the code-only figure-pipeline branch.

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
- [**Metrics →**]({{< ref "/metrics" >}}) 20 publication-display numeric
  metrics: 3 clustering-compactness scores, 10 DRE neighbourhood/scale
  diagnostics, and 7 latent-space intrinsic-geometry scores, with two retained
  intrinsic annotations documented separately.
- [**About →**]({{< ref "/about" >}}) Paper citation, reproducibility pointers,
  license.
