---
title: "GATv2"
type: docs
weight: 101
geekdocHidden: false
---

# GATv2

| Field | Value |
|-------|-------|
| **Family** | attention |
| **Group** | scCCVGBen graph encoder |

## Description

Dynamic attention Graph Attention v2 (Brody 2022) — scCCVGBen extension

## Role in scCCVGBen

Axis A (encoder-variation) sweep: CCVGAE trains a latent representation
with this message-passing / attention module while holding the graph fixed
to k-NN Euclidean. Benchmark naming for sweep rows: `scCCVGBen_GATv2`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and 26 metrics; see the
[Methods index](../) for the full set.</small>
