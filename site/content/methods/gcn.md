---
title: "GCN"
type: docs
weight: 104
geekdocHidden: false
---

# GCN

| Field | Value |
|-------|-------|
| **Family** | message-pass |
| **Group** | scCCVGBen graph encoder |

## Description

Graph Convolutional Network (Kipf 2017)

## Role in scCCVGBen

Axis A (encoder-variation) sweep: scCCVGBen trains a latent representation
with this message-passing / attention module while holding the graph fixed
to k-NN Euclidean. Benchmark naming for sweep rows: `scCCVGBen_GCN`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and curated 20 publication-display
metrics; see the [Methods index](../) for the full set.</small>
