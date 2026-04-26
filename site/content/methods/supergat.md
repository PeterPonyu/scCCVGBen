---
title: "SuperGAT"
type: docs
weight: 103
geekdocHidden: false
---

# SuperGAT

| Field | Value |
|-------|-------|
| **Family** | attention |
| **Group** | scCCVGBen graph encoder |

## Description

Self-supervised edge prediction GAT (Kim 2020) — scCCVGBen extension

## Role in scCCVGBen

Axis A (encoder-variation) sweep: scCCVGBen trains a latent representation
with this message-passing / attention module while holding the graph fixed
to k-NN Euclidean. Benchmark naming for sweep rows: `scCCVGBen_SuperGAT`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and curated 20 publication-display
metrics; see the [Methods index](../) for the full set.</small>
