---
title: "gaussian_threshold"
type: docs
weight: 118
geekdocHidden: false
---

# gaussian_threshold

| Field | Value |
|-------|-------|
| **Family** | graph |
| **Group** | Graph construction method |

## Description

Gaussian heat-kernel weights w=exp(-d²/(2σ²)); edges pruned at threshold 0.9

## Role in scCCVGBen

Axis B (graph-construction sweep): scCCVGBen encoder is fixed to GAT while
this graph builder constructs the cell-cell neighbourhood fed to the
encoder. Benchmark naming: `scCCVGBen_GAT_gaussian_threshold`.


---

<small>Auto-generated from `scripts/build_site_data.py`. scCCVGBen benchmark
tests each method against the same 200 datasets and curated 20 publication-display
metrics; see the [Methods index](../) for the full set.</small>
