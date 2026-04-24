# scCCVGBen

**Single-cell CCVGAE Benchmark** — an independent benchmark suite extending the CCVGAE method
(Centroid-based Coupled Variational Graph Attention AutoEncoder) across 200 single-cell omics
datasets (100 scRNA + 100 scATAC).

**Repo location**: `/home/zeyufu/LAB/scCCVGBen/` — this is an **independent repo**, fully
separate from `/home/zeyufu/LAB/CCVGAE/`. Data and pre-computed results are reused via
workspace symlinks (see `data/reuse_map.csv`); no shared Python imports exist between the two.

---

## Mission

scCCVGBen delivers three computational-experiment axes on a rebalanced 100 scRNA + 100 scATAC
benchmark of 200 datasets:

### Axis A — Encoder Validation
12 graph encoders × fixed kNN-Euclidean (k=15) graph × 200 datasets = 2,400 runs.

**Hypothesis**: attention-class encoders (GAT, GATv2, TransformerConv, SuperGAT) outperform
the 8 message-passing encoders (GCN, GraphSAGE, GIN, ChebNet, EdgeConv, ARMAConv, SGConv,
TAGConv) on the CCVGAE objective across a majority of datasets.

### Axis B — Graph-Construction Validation
Fixed GAT encoder × 5 diverse graph constructions (kNN-Euc, kNN-cos, SNN, Mutual-kNN,
Gaussian-threshold) × 200 datasets = 1,000 runs.

**Hypothesis**: CCVGAE is robust to graph construction choice — low coefficient of variation
on primary metrics across all 5 graph types.

### Axis C — Baseline Benchmarking
13 baselines (PCA, KPCA, ICA, FA, NMF, TSVD, DICL, scVI, DIPVAE, InfoVAE, TCVAE,
HighBetaVAE, plus reference CCVGAE) × 200 datasets = ~2,600 runs.

**Hypothesis**: CCVGAE (best attention-class encoder + any of the 5 graph constructions)
beats all baselines on a majority (≥101/200) of datasets across primary metrics (ARI, NMI,
ASW, distance-correlation), as validated by paired Wilcoxon signed-rank test (p<0.05,
Holm-Bonferroni corrected) with Cliff's delta effect size.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd scCCVGBen
pip install -e ".[dev]"

# 2. Establish workspace symlinks to prior CCVGAE data/results
#    (requires /home/zeyufu/LAB/CCVGAE/ on the same machine)
bash scripts/setup_symlinks.sh

# 3. Verify imports
pytest tests/test_imports.py

# 4. See LAUNCH_BENCHMARK.md for the full sweep
```

---

## Repository Structure

```
scCCVGBen/
├── scccvgben/          # Python package (models, graphs, training, stats, baselines)
├── scripts/            # Data curation + sweep runner scripts
├── results/            # Committed CSV results (small files)
├── data/               # Committed metadata CSVs (datasets.csv, reuse_map.csv, dropped_scatac_v2.csv)
├── figures/            # Committed publication figures
├── site/               # Hugo static site (dataset cards + filterable index)
├── tests/              # Smoke tests
└── workspace/          # GITIGNORED — symlinked data, cached results, checkpoints
```

---

## References

- CCVGAE paper: [placeholder — link to published article]
- Benchmark spec: `/home/zeyufu/LAB/CCVGAE/.omc/specs/deep-interview-scccvgben-v2.md`
- Implementation plan: `/home/zeyufu/LAB/CCVGAE/.omc/plans/autopilot-impl.md`

---

## License

MIT — see [LICENSE](LICENSE).
