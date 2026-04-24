---
title: "Cheb"
type: docs
weight: 107
geekdocHidden: false
---

# Cheb

| Field | Value |
|-------|-------|
| **Family** | message-pass |
| **Group** | scCCVGBen graph encoder |

## Description

ChebNet — Chebyshev polynomial filters (Defferrard 2016)

## Role in scCCVGBen

Axis A (encoder-variation) sweep: CCVGAE trains a latent representation
with this message-passing / attention module while holding the graph fixed
to k-NN Euclidean. Benchmark naming for sweep rows: `scCCVGBen_Cheb`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and 26 metrics; see the
[Methods index](../) for the full set.</small>
