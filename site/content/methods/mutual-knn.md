---
title: "mutual_knn"
type: docs
weight: 117
geekdocHidden: false
---

# mutual_knn

| Field | Value |
|-------|-------|
| **Family** | graph |
| **Group** | Graph construction method |

## Description

Mutual k-NN — only edges where both cells are in each other's k-NN list; stricter connectivity

## Role in scCCVGBen

Axis B (graph-construction sweep): scCCVGBen encoder is fixed to GAT while
this graph builder constructs the cell-cell neighbourhood fed to the
encoder. Benchmark naming: `scCCVGBen_GAT_mutual_knn`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and 26 metrics; see the
[Methods index](../) for the full set.</small>
