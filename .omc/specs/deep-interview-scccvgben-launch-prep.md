# Deep Interview Spec: scCCVGBen v0.1.0 — Launch Preparation Session

## Metadata

- **Interview ID**: scccvgben-launch-prep
- **Rounds**: 4
- **Final Ambiguity Score**: 8.05%
- **Type**: brownfield
- **Generated**: 2026-04-23
- **Threshold**: 20% (default; no project override)
- **Status**: PASSED (all dimensions ≥ 0.85)
- **Repo**: `/home/zeyufu/LAB/scCCVGBen/`
- **Upstream Spec (benchmark-level)**: `/home/zeyufu/LAB/CCVGAE/.omc/specs/deep-interview-scccvgben-v2.md`

## Clarity Breakdown

| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.95 | 0.35 | 0.3325 |
| Constraint Clarity | 0.90 | 0.25 | 0.2250 |
| Success Criteria | 0.92 | 0.25 | 0.2300 |
| Context Clarity | 0.88 | 0.15 | 0.1320 |
| **Total Clarity** | | | **0.9195** |
| **Ambiguity** | | | **0.0805** |

---

## Goal

Prepare the `scCCVGBen` v0.1.0 repository so that the next session can launch a
full 200-dataset GPU sweep **without any blocking gaps**. This session does NOT
run the GPU sweep itself. It delivers:

1. All 9 `KNOWN_ISSUES.md` items fixed ("死死全收" — no partial delivery).
2. `fetch_geo_scrna.py` promoted from skeleton to a real, runnable downloader
   using a GEO proxy API, fetching `filtered_feature_matrices`-class `.h5`
   files (raw unnormalized counts, moderate size, no atlas-scale files).
3. A complete data inventory: revised-methodology-aligned manifest of what
   exists on-host, reconciled against what revised CCVGAE actually needs.
4. The GEO download executed to fill any residual data gap.
5. A handoff document recording every modification so the next session can
   pick up seamlessly.

**Terminal deliverable of the benchmark** (informs every detail above): paper
submission + public Hugo leaderboard site.

## Constraints

- **Methodology anchor**: revised CCVGAE (post-review). Authoritative code
  location: `/home/zeyufu/LAB/CCVGAE/CCVGAE1_SD.ipynb`,
  `CCVGAE2_UCB.ipynb`, `CCVGAE3_IR.ipynb` (updated 2026-04-23) plus
  `/home/zeyufu/LAB/CCVGAE/.omc/specs/` and the review-response diff at
  `CCVGAE_snLaTeX_diff/`. The **initial** CCVGAE version must not be used as
  the methodological reference.
- **Data-format gate for GEO downloads**:
  - `filtered_feature_bc_matrix`-class `.h5` (or equivalent h5ad of raw counts).
  - Raw, unnormalized expression matrices only.
  - Moderate size — reject atlas-scale datasets (e.g., HCL-style million-cell
    objects) in favor of datasets easily loaded into memory.
  - Preserve key metadata (`sample`, `condition`, `tissue`, `organism`,
    `celltype` where published).
- **Downloader design**: use a GEO proxy API (GEOparse's `get_GEO` or the NCBI
  E-utilities via the HTTPS proxy) — **not** direct FTP.
- **Work ordering** (inventory-first): audit revised methodology + scan
  `~/Downloads/` and all known data directories → update
  `existing_scrna_diversity.csv` and the scATAC on-host manifest → re-compute
  the true GEO gap → auto-select candidates via diversity rubric for only
  the residual gap. Minimize redundant downloads.
- **Bug fix scope**: all 9 items in `KNOWN_ISSUES.md`, this session.
- **No GPU execution this session**: benchmark runtime (~2-3 days on one
  local GPU, ~1 min/run) is scheduled for a later session.
- **Publication-grade rigor** (implied by paper+site deliverable):
  - GAT-row provenance asymmetry must be properly disclosed (in
    `results/encoder_sweep/README.md` plus paper methods).
  - Paired Wilcoxon signed-rank + Holm-Bonferroni + Cliff's delta will be
    required in the later GPU-sweep session.
  - Figures must meet publication standards (300+ dpi, vector PDF) in the
    later session.
  - Hugo site must be reproducible and deployable.

## Non-Goals (this session)

- Running any axis sweep (Axis A / B / C). Smoke runs only for validation.
- Generating publication figures.
- Building or deploying the Hugo site (Phase 3 work, next session).
- Writing the paper manuscript.
- Running statistical tests on result tables.
- Redesigning the 200-dataset benchmark size (100 scRNA + 100 scATAC is
  inherited from the upstream spec and remains the target).

## Acceptance Criteria

- [ ] **AC-1** `pytest tests/ -q` remains green (≥ 31/31 passing; new
  regression tests may raise the count) before and after every fix.
- [ ] **AC-2** `KNOWN_ISSUES.md` #1 (metrics binning `duplicates="drop"`) fixed;
  synthetic and a real-h5ad smoke run both yield non-empty metric columns.
- [ ] **AC-3** `KNOWN_ISSUES.md` #2 (GATv2 silent NaN) fixed; GATv2 row in
  `results/encoder_sweep/synthetic_200x100.csv` contains real numbers.
- [ ] **AC-4** `KNOWN_ISSUES.md` #3-4 (scATAC h5ad source located, drop
  manifest `cell_count` populated for all 15 rows) resolved; source path is
  added to `SCRNA_SOURCES` in `scripts/setup_symlinks.sh` (or a parallel
  `SCATAC_SOURCES`), and `setup_symlinks.sh` reports `h5ad scATAC: N>0`.
- [ ] **AC-5** `KNOWN_ISSUES.md` #5-8 (dead import, DRY label extraction,
  GH Actions SHA pin, `setup_symlinks.sh` path parameterization) all done.
- [ ] **AC-6** `KNOWN_ISSUES.md` #9 fully resolved — `fetch_geo_scrna.py` is a
  real downloader (no `NotImplementedError`), consumes a GEO proxy API,
  honors `--target`, `--candidate-pool`, `--out`, and `--candidate-csv`,
  is idempotent on re-run, writes proxy download logs to
  `workspace/logs/fetch_geo.log`, and produces valid `.h5ad` files with raw
  counts + metadata under `workspace/data/scrna/`.
- [ ] **AC-7** Revised-methodology data manifest built: a new script or
  updated `scripts/build_datasets_csv.py` reads the revised CCVGAE notebooks
  (or the upstream `.omc/specs/`) and produces a canonical "what-we-need"
  list, which is then reconciled against what is actually on-host.
- [ ] **AC-8** Complete host inventory written to
  `data/existing_scrna_diversity.csv` (currently 27 rows) and a new
  `data/existing_scatac_inventory.csv`, each row including absolute path,
  GSE, tissue, organism, cell_count, provenance folder.
- [ ] **AC-9** `data/scrna_candidate_pool.csv` expanded to ≥ 60 candidate GSEs,
  auto-curated by the diversity rubric from the upstream spec, covering the
  residual gap left after the host inventory. Curation script + rubric is
  reproducible and committed.
- [ ] **AC-10** `fetch_geo_scrna.py` executed to close the real gap. At the
  end, `scripts/verify_benchmark_size.py` prints 100 scRNA + 100 scATAC = 200
  (or, if fewer total datasets are justified by revised methodology, the
  agreed revised count with a one-line justification in the handoff doc).
- [ ] **AC-11** A handoff document at `HANDOFF_GPU_SESSION.md` (or
  equivalent) enumerates every file changed, every new script added, every
  assumption made, every GSE queued for download (plus any that failed),
  and an explicit "to run the sweep, execute …" checklist that maps onto
  `LAUNCH_BENCHMARK.md` STAGE 2–3.
- [ ] **AC-12** All changes committed in focused, reviewable commits
  (one commit per KNOWN_ISSUES item or coherent logical group).
  Pre-commit / CI hooks honored.

## Assumptions Exposed & Resolved

| Assumption | Challenge | Resolution |
|---|---|---|
| Original plan was "fetch 45 new GEOs blindly" | Revised CCVGAE may use different data; ~/Downloads (129 GB) may already cover gaps | **Inventory-first**: audit revised methodology + on-host data, then fetch only the residual. |
| "Fix high-priority only" might be sufficient | Paper+site deliverable demands a clean repo | **Fix all 9** this session. |
| Terminal deliverable is internal | User is pursuing publication AND public leaderboard | Paper+site — provenance asymmetry, stats rigor, figure quality all mandated downstream. |
| GPU budget is the bottleneck | 1 min/run, 2-3 days total on local GPU is acceptable | **Not** a bottleneck; next session handles it. |
| Methodological reference is the original CCVGAE code | User corrected: revised version (2026-04-23) is the anchor | Read `CCVGAE1_SD.ipynb`, `CCVGAE2_UCB.ipynb`, `CCVGAE3_IR.ipynb` and `CCVGAE_snLaTeX_diff/` before deciding on data manifest. |
| 60-GSE candidate pool is fixed | Host inventory may shrink the real need | Expand pool to ≥ 60 **after** audit; actual download target = residual gap. |

## Technical Context

### Repository state (as of interview)
- **Files**: 82 (2 commits: scaffold + Phase 5 docs).
- **Tests**: 31/31 passing (28 unit + 3 integration).
- **Reuse symlinks**: 225 (55 scrna_baselines + 55 axisA_GAT_scrna +
  115 scatac_baselines) + 46 scRNA `.h5ad` symlinks.
- **Committed CSVs** (in `data/`): `reuse_map.csv` (5 rows),
  `scrna_candidate_pool.csv` (10 rows, needs expansion),
  `existing_scrna_diversity.csv` (27 rows),
  `dropped_scatac_v2.csv` (15 rows, `cell_count` blank),
  `scatac_baseline_method_list.csv` (4 rows).
- **Skeleton scripts** (confirmed): `scripts/fetch_geo_scrna.py` line 78
  raises `NotImplementedError`.

### Revised methodology anchor
- `/home/zeyufu/LAB/CCVGAE/CCVGAE1_SD.ipynb` (3.2 MB, 2026-04-23)
- `/home/zeyufu/LAB/CCVGAE/CCVGAE2_UCB.ipynb` (4.4 MB, 2026-04-23)
- `/home/zeyufu/LAB/CCVGAE/CCVGAE3_IR.ipynb` (3.9 MB, 2026-04-23)
- `/home/zeyufu/LAB/CCVGAE/.omc/specs/` (revised specs)
- `/home/zeyufu/LAB/CCVGAE/CCVGAE_snLaTeX_diff/` (review-response diff)
- `/home/zeyufu/LAB/CCVGAE/CCVGAE_review_pdfs/` (reviewer feedback)

### On-host data pool (preliminary — must be audited in execution)
- `~/Downloads/` (~129 GB total), organized into:
  - Top-level: `hESC_GSE144024.h5ad`, `adata.h5ad`
  - `~/Downloads/DevelopmentDatasets/` ≈ 15 files
  - `~/Downloads/CancerDatasets/` ≥ 11 files
- Already symlinked into workspace: 46 scRNA `.h5ad`.
- `/home/zeyufu/LAB/DATA/`, `/home/zeyufu/LAB/SCRL/`,
  `/home/zeyufu/vGAE_LAB/data/` — additional scRNA source dirs.
- scATAC on host: **0 h5ad located via current SCRNA_SOURCES**. Likely in
  one of the LAB subdirs (CODEVAE/LIVAE/MCOGVAE/etc.) or the Downloads tree
  — to be traced via revised CCVGAE code's scATAC loader.

### Upstream reuse map (not to be modified)
- `/home/zeyufu/LAB/CCVGAE/CG_results/CG_dl_merged/` → 55 scRNA × 13 baselines.
- `/home/zeyufu/LAB/CCVGAE/CG_results/CG_atacs/tables/` → 115 scATAC baselines.
- `archived_extra_scATAC/` in `CG_results/` — use filenames as a hint for
  scATAC source location (per `KNOWN_ISSUES.md` #3 action note).

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|---|---|---|---|
| Benchmark | core | 3 axes, 200 datasets, 3 metrics family | produces Paper, Public Site |
| Paper | core deliverable | venue TBD, methods, figures | depends on Benchmark results + Statistical Test |
| Public Site | core deliverable | Hugo static, dataset cards, filterable index | depends on Dataset Meta Data |
| CCVGAE Method | supporting | encoder, graph, latent, losses | benchmarked by Axis A/B/C |
| Revised CCVGAE Methodology | core | 3 notebooks (SD/UCB/IR), review diff | authoritative reference for Dataset selection |
| Axis A | supporting | 12 encoders × 1 graph × 200 datasets | produces encoder ranking |
| Axis B | supporting | 1 encoder × 5 graphs × 200 datasets | tests graph robustness |
| Axis C | supporting | 13 baselines × 200 datasets | supports CCVGAE superiority claim |
| Encoder | supporting | type (GAT/GATv2/GCN/...), kwargs | member of Axis A |
| Graph Construction | supporting | kNN-Euc/kNN-cos/SNN/Mutual-kNN/Gaussian | member of Axis B |
| Baseline | supporting | PCA/scVI/DIPVAE/... | member of Axis C |
| Dataset | core | GSE, tissue, organism, cell_count, modality | row in datasets.csv |
| Downloads Data Pool | core | ~129 GB on host, 3+ subfolders | source for on-host Dataset inventory |
| Handoff Document | core | file list, GSE queue, checklist | bridges this session → GPU-sweep session |
| Data Integration | core | on-host ∪ newly-fetched → canonical datasets.csv | output of inventory + fetch |
| Meta Data | supporting | per-dataset metadata (tissue/organism/etc.) | feeds Public Site cards |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Removed | Stability |
|-------|-------------|-----|---------|--------|---------|-----------|
| 1 | 12 | 12 | - | - | - | N/A |
| 2 | 13 | 3 | 0 | 10 | 2 | 77% |
| 3 | 14 | 2 | 0 | 12 | 1 | 86% |
| 4 | 14 | 0 | 0 | 14 | 0 | **100%** |

Domain model converged at Round 4 — stable entity set.

## Interview Transcript

<details>
<summary>Full Q&A (4 rounds)</summary>

### Round 1 — Targeting: Success Criteria (0.30)
**Q:** scCCVGBen 的「完成」指什么？(论文 / 内部基线 / 公开站点 / 多目标)
**A:** 多目标：论文 + 公开站点
**Ambiguity after:** 28.7% (Goal 0.78, Constraints 0.45, Criteria 0.80, Context 0.85)

### Round 2 — Targeting: Constraints (0.45)
**Q:** GPU + 时间预算？
**A:** 硬件非瓶颈，2-3 天本机 GPU 可完成；**本 session 不跑 GPU**，只做脚本完善 + 修 bug + 下数据 + 记录修改过程，确保无缝衔接下一阶段的计算运行、两阶段数据整合、论文构建、网站搭建。
**Ambiguity after:** 12.55% (Goal 0.92, Constraints 0.85, Criteria 0.85, Context 0.85)

### Round 3 — Targeting: residual Goal + Context gaps
**Q:** GEO 池填补策略 + Bug 修复范围？
**A:**
- GEO 池：自动筛选（diversity rubric）
- Bug 范围：**同时勾选了 "4 项" 和 "9 项"** — 矛盾，待 Round 4 澄清
- 额外指示：使用 revised 版 CCVGAE 训练代码作为锚点（非初始版本）；数据在 `~/Downloads/`；GEO 用代理 API；要 filtered_feature_matrices 类 h5（raw counts，中等体积，非图谱）
**Ambiguity after:** 13.95% (Goal 0.88, Constraints 0.85, Criteria 0.88, Context 0.80)

### Round 4 — Disambiguating Bug scope + Data plan
**Q1:** Bug 范围真实意图？
**A1:** **全修 9 项，本 session 死死全收**
**Q2:** 下一步做什么？
**A2:** **先 inventory，再决定 GEO 缺口**（最小驱动原则）
**Ambiguity after:** 8.05% (Goal 0.95, Constraints 0.90, Criteria 0.92, Context 0.88) — **PASSED**

</details>

---

## Execution Plan (normative work order)

1. **Read revised methodology** — extract data manifest from
   `CCVGAE1_SD/2_UCB/3_IR.ipynb` + `CCVGAE/.omc/specs/` + `CCVGAE_snLaTeX_diff/`.
2. **Host inventory** — scan `~/Downloads/`, `/home/zeyufu/LAB/DATA/`,
   `/home/zeyufu/LAB/SCRL/`, `/home/zeyufu/vGAE_LAB/data/`, and LAB subdirs
   for both scRNA and scATAC `.h5ad`/`.h5`. Extract GSE, tissue, organism,
   cell_count, modality. Write to
   `data/existing_scrna_diversity.csv` (expand from 27) and
   `data/existing_scatac_inventory.csv` (new).
3. **Fix KNOWN_ISSUES #3 + #4** as a byproduct of step 2 (scATAC source
   located + drop manifest `cell_count` filled).
4. **Fix KNOWN_ISSUES #1** — `scccvgben/training/metrics.py` binning
   `duplicates="drop"`. Add a regression test with degenerate latent input.
5. **Fix KNOWN_ISSUES #2** — `scccvgben/models/encoder_registry.py` GATv2
   kwargs. Add a regression test that asserts GATv2 produces finite outputs
   on synthetic input.
6. **Fix KNOWN_ISSUES #5-8** — dead import, DRY label extraction,
   GH Actions SHA pin, `setup_symlinks.sh` path parameterization.
7. **Fix KNOWN_ISSUES #9** — implement real GEO downloader in
   `scripts/fetch_geo_scrna.py` using GEOparse or E-utilities via proxy,
   filtering for `filtered_feature_bc_matrix`-class outputs, preserving
   metadata, and writing `.h5ad` files to `workspace/data/scrna/`. Add a
   dry-run test that stubs the proxy and checks idempotency.
8. **Reconcile revised manifest vs. host inventory** → compute residual gap.
9. **Auto-curate candidate pool** — expand `data/scrna_candidate_pool.csv`
   to ≥ 60 entries by applying the diversity rubric from the upstream spec.
   Commit the curation script.
10. **Execute real download** for the residual gap → raw-count `.h5ad`s in
    `workspace/data/scrna/`. Persist logs to `workspace/logs/`.
11. **Rebuild datasets.csv** (`scripts/build_datasets_csv.py --build-canonical`)
    so it reflects the full 200-dataset (or revised-count) benchmark.
12. **Verify size** (`scripts/verify_benchmark_size.py`) — must print the
    agreed total.
13. **Run smoke tests** on a handful of real h5ads to confirm bug fixes and
    data loading. Do NOT run the full sweep.
14. **Write `HANDOFF_GPU_SESSION.md`** — file list, assumption log, GSE
    queue (downloaded + failed), and an explicit STAGE 2 launch checklist.
15. **Commit in focused groups**; ensure `pytest tests/` green at each commit.

## Handoff to Execution

This spec is ready for consumption by `omc-plan --consensus --direct` →
`autopilot`, OR direct execution via `autopilot` / `ralph` / `team` /
`ultrawork`. The user will select via the execution-bridge question that
follows.
