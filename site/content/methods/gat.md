---
title: "GAT"
type: docs
weight: 100
geekdocHidden: false
---

# GAT

| Field | Value |
|-------|-------|
| **Family** | attention |
| **Group** | scCCVGBen graph encoder |

## Description

Graph Attention Network (Veličković 2018)

## Role in scCCVGBen

Axis A (encoder-variation) sweep: scCCVGBen trains a latent representation
with this message-passing / attention module while holding the graph fixed
to k-NN Euclidean. Benchmark naming for sweep rows: `scCCVGBen_GAT`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and 26 metrics; see the
[Methods index](../) for the full set.</small>
