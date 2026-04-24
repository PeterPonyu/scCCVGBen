---
title: "kNN_euclidean"
type: docs
weight: 114
geekdocHidden: false
---

# kNN_euclidean

| Field | Value |
|-------|-------|
| **Family** | graph |
| **Group** | Graph construction method |

## Description

Standard k-NN with Euclidean distance (k=15) — CCVGAE benchmark default

## Role in scCCVGBen

Axis B (graph-construction sweep): CCVGAE encoder is fixed to GAT while
this graph builder constructs the cell-cell neighbourhood fed to the
encoder. Benchmark naming: `scCCVGBen_GAT_kNN_euclidean`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and 26 metrics; see the
[Methods index](../) for the full set.</small>
