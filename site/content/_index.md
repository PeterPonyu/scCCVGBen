---
title: "scCCVGBen"
type: docs
---

# scCCVGBen Benchmark Atlas

**scCCVGBen** links the curated benchmark roster to reproducible, paper-ready
result panels.

---

## Benchmark at a glance

{{< headline-stats >}}

---

## Benchmark composition

{{< home-composition >}}

---

## Model architecture

Figure 2 summarizes the model-first path through scCCVGBen: input preprocessing,
five graph-construction choices, the 14-encoder registry, variational latent
flow, dual reconstruction outputs, and the grouped 20-display-metric evaluation
table.
The publication image is rendered by the reproducible figure pipeline; generated binaries are refreshed from the curated data artifacts.

---

## Metadata distributions

{{< extended-stats >}}

---

## Explore

- [**Datasets →**]({{< ref "/datasets" >}}) Browse all 200 benchmark datasets
  with per-dataset detail pages (manifest metadata, PubMed links, GEO accessions,
  and public-safe redaction notes where applicable).
- [**Methods →**]({{< ref "/methods" >}}) 14 scCCVGBen graph encoders
  (attention + message-passing), 5 graph construction methods, and 13
  comparator entries (12 external baselines plus the scCCVGBen reference row).
- [**Metrics →**]({{< ref "/metrics" >}}) 20 publication-display numeric
  metrics: 3 clustering-compactness scores, 10 DRE neighbourhood/scale
  diagnostics, and 7 latent-space intrinsic-geometry scores, with two retained
  restricted-access record handling documented separately.
- [**About →**]({{< ref "/about" >}}) Paper citation, reproducibility pointers,
  license.

- [**Next.js companion explorer →**](https://peterponyu.github.io/scccvgben-next/) Open the responsive companion browser for the same benchmark roster. It is published separately from this Hugo atlas, so `/scCCVGBen/` remains the documentation atlas while `/scccvgben-next/` remains the interactive companion route.
- [**scPortal discovery hub →**](https://peterponyu.github.io/scportal/) Navigate the broader public graph that links the homepage, scPortal, this Hugo atlas, and the Next.js companion.
