---
title: "kNN_cosine"
type: docs
weight: 115
geekdocHidden: false
---

# kNN_cosine

| Field | Value |
|-------|-------|
| **Family** | graph |
| **Group** | Graph construction method |

## Description

k-NN with cosine similarity — rewards direction, invariant to magnitude

## Role in scCCVGBen

Axis B (graph-construction sweep): scCCVGBen encoder is fixed to GAT while
this graph builder constructs the cell-cell neighbourhood fed to the
encoder. Benchmark naming: `scCCVGBen_GAT_kNN_cosine`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and curated 20 publication-display
metrics; see the [Methods index](../) for the full set.</small>
