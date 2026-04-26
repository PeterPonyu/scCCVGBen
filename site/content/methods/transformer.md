---
title: "Transformer"
type: docs
weight: 102
geekdocHidden: false
---

# Transformer

| Field | Value |
|-------|-------|
| **Family** | attention |
| **Group** | scCCVGBen graph encoder |

## Description

TransformerConv — Attention is All You Need graph variant (Shi 2020)

## Role in scCCVGBen

Axis A (encoder-variation) sweep: scCCVGBen trains a latent representation
with this message-passing / attention module while holding the graph fixed
to k-NN Euclidean. Benchmark naming for sweep rows: `scCCVGBen_Transformer`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and curated 20 publication-display
metrics; see the [Methods index](../) for the full set.</small>
