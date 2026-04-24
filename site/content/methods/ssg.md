---
title: "SSG"
type: docs
weight: 111
geekdocHidden: false
---

# SSG

| Field | Value |
|-------|-------|
| **Family** | message-pass |
| **Group** | scCCVGBen graph encoder |

## Description

Simple Spectral Graph Conv (Zhu 2021)

## Role in scCCVGBen

Axis A (encoder-variation) sweep: CCVGAE trains a latent representation
with this message-passing / attention module while holding the graph fixed
to k-NN Euclidean. Benchmark naming for sweep rows: `scCCVGBen_SSG`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and 26 metrics; see the
[Methods index](../) for the full set.</small>
