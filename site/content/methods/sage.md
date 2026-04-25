---
title: "SAGE"
type: docs
weight: 105
geekdocHidden: false
---

# SAGE

| Field | Value |
|-------|-------|
| **Family** | message-pass |
| **Group** | scCCVGBen graph encoder |

## Description

GraphSAGE inductive (Hamilton 2017)

## Role in scCCVGBen

Axis A (encoder-variation) sweep: scCCVGBen trains a latent representation
with this message-passing / attention module while holding the graph fixed
to k-NN Euclidean. Benchmark naming for sweep rows: `scCCVGBen_SAGE`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and 26 metrics; see the
[Methods index](../) for the full set.</small>
