---
title: "GIN"
type: docs
weight: 112
geekdocHidden: false
---

# GIN

| Field | Value |
|-------|-------|
| **Family** | message-pass |
| **Group** | scCCVGBen graph encoder |

## Description

Graph Isomorphism Network (Xu 2019) — scCCVGBen extension

## Role in scCCVGBen

Axis A (encoder-variation) sweep: CCVGAE trains a latent representation
with this message-passing / attention module while holding the graph fixed
to k-NN Euclidean. Benchmark naming for sweep rows: `scCCVGBen_GIN`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and 26 metrics; see the
[Methods index](../) for the full set.</small>
