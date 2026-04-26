---
title: "snn"
type: docs
weight: 116
geekdocHidden: false
---

# snn

| Field | Value |
|-------|-------|
| **Family** | graph |
| **Group** | Graph construction method |

## Description

Shared Nearest Neighbour — 2 cells connected if they share a fraction of neighbours

## Role in scCCVGBen

Axis B (graph-construction sweep): scCCVGBen encoder is fixed to GAT while
this graph builder constructs the cell-cell neighbourhood fed to the
encoder. Benchmark naming: `scCCVGBen_GAT_snn`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and curated 20 publication-display
metrics; see the [Methods index](../) for the full set.</small>
