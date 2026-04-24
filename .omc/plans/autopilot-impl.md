# scCCVGBen Autopilot Phase 2 Implementation Plan

- **Spec**: `/home/zeyufu/LAB/scCCVGBen/.omc/specs/deep-interview-scccvgben-launch-prep.md`
- **Repo root**: `/home/zeyufu/LAB/scCCVGBen/`
- **Generated**: 2026-04-23 (Architect, Phase 1)
- **Target ambiguity ceiling**: 8.05% (already PASSED in interview)
- **Output consumer**: Phase 2 executor(s) — split by agent tier (haiku/sonnet/opus)

---

## 1. 概览

### 1.1 目标（单句）
把 repo 从"scaffold + 9 个已知 bug"推进到"一条命令就能在 GPU session 跑完 200-数据集 sweep"的就绪态，**本 session 不触发 GPU**。

### 1.2 硬约束摘要（不可动）
1. 9 项 `KNOWN_ISSUES` 全修（死死全收，无部分交付）。
2. GPU 不跑（仅 smoke 验证），但 inventory / 下载 / 指标计算必须真实跑过。
3. Inventory-first：revised methodology manifest → on-host 扫描 → 残差 gap → GEO 下载。
4. GEO 下载必须走**代理 API**（GEOparse 或 E-utilities via HTTPS proxy），**禁止**直接 FTP。
5. 只收 `filtered_feature_bc_matrix` 类 h5 / 等价 raw-count h5ad，**拒** atlas 级。
6. 方法论锚点 = revised CCVGAE (`/home/zeyufu/LAB/CCVGAE/CCVGAE{1_SD,2_UCB,3_IR}.ipynb`, 2026-04-23)，**非**初始版本。
7. `pytest tests/ -q` 每次 commit 后仍需 ≥ 31/31 绿色。
8. 默认 200 数据集 (100 scRNA + 100 scATAC)；如 revised methodology 指向不同数量，按 revised 并在 handoff 注明一行 justification。

### 1.3 预计交付时长（人时估算，含调试/重跑）
| 段 | 内容 | 估时 |
|---|---|---|
| Bug-fix 块 (T4–T8, T10) | 6 个 code fix + 回归测试 | 3–4 h |
| Inventory 块 (T1–T3, T9) | notebook 解析 + 扫盘 + CSV 更新 + 缺口计算 | 2–3 h |
| Downloader 块 (T11–T13) | GEOparse wrapper + candidate pool 扩张 + 真实下载 | **4–10 h**（受网络/限速支配） |
| Integration 块 (T14–T16) | dataset.csv 重建 + verify + smoke | 1–2 h |
| Handoff + commits (T17–T18) | 写 HANDOFF md + 分组提交 | 1–1.5 h |
| **合计** | | **11–20.5 人时**；网络快 + 无意外 ≈ 12 h，限速严重可达 20 h |

### 1.4 关键路径（决定总时长）
```
T1 (read revised notebooks)
  └─> T2 (on-host inventory scan)
        ├─> T3 (scATAC source + drop manifest cell_count)   [AC-4, AC-8 partial]
        └─> T9 (reconcile → residual gap)
              └─> T11 (expand candidate pool ≥60)           [AC-9]
                    └─> T12 (real GEO download)             [AC-6, AC-10]  ← 时长瓶颈
                          └─> T14 (rebuild datasets.csv)
                                └─> T15 (verify_benchmark_size)
                                      └─> T16 (smoke on real h5ad)
                                            └─> T17 (HANDOFF doc)
                                                  └─> T18 (commits)
```
Bug-fix 侧链 (T4–T8, T10) 可与 T1–T3 完全并行，**不**在关键路径。

---

## 2. 工作分解

每个任务单元格式：
- **ID / 标题**
- **Agent tier**（haiku = 机械重复 / sonnet = 标准实现 / opus = 判断密集 / 需 NotebookRead / 需跨仓推理）
- **输入**
- **输出**
- **依赖**
- **验收映射**（→ spec AC-N）
- **做法摘要**（executor 的"下一步指令"）

### T1. 提取 revised CCVGAE 数据 manifest
- **Tier**: opus （notebook 代码阅读 + 跨 3 个 notebook 的 manifest 合并）
- **输入**:
  - `/home/zeyufu/LAB/CCVGAE/CCVGAE1_SD.ipynb`
  - `/home/zeyufu/LAB/CCVGAE/CCVGAE2_UCB.ipynb`
  - `/home/zeyufu/LAB/CCVGAE/CCVGAE3_IR.ipynb`
  - `/home/zeyufu/LAB/CCVGAE/.omc/specs/` (revised specs)
  - `/home/zeyufu/LAB/CCVGAE/CCVGAE_snLaTeX_diff/` (review-response diff)
- **输出**:
  - `data/revised_methodology_manifest.csv` — 列: `modality,expected_count,tissue,organism,source_file_path,notes`
  - `.omc/plans/revised-methodology-notes.md` — 对 200 vs 其它数量的判定依据
- **依赖**: 无（入口）
- **验收**: AC-7
- **做法**: 用 `jupyter nbconvert --to script` 或直接 NotebookRead 每个 notebook 的 data-loading cells；grep `h5ad` / `GSE` / `load` / `pd.read` / `sc.read` 模式；合并去重后写 CSV；在 `revised-methodology-notes.md` 明确回答"revised methodology 要求 100+100=200 吗？"—若有差异必须一行 justification。

### T2. On-host h5ad inventory 全量扫描
- **Tier**: sonnet
- **输入**:
  - `~/Downloads/` (top + DevelopmentDatasets/ + DevelopmentDatasets2/ + CancerDatasets/ + CancerDatasets2/ + ATAC_data/)
  - `/home/zeyufu/LAB/DATA/`, `/home/zeyufu/LAB/SCRL/`, `/home/zeyufu/vGAE_LAB/data/`
  - `/home/zeyufu/LAB/MCC_results/`, `/home/zeyufu/Desktop/{HMJ,CSD,WC,LSK-LCN-Publicdata}/`
  - `/home/zeyufu/LAB/processed_datasets/`, LAB 其它 subdir（CODEVAE/LIVAE/MCOGVAE/IVAE/…）的 `.h5ad`
- **输出**:
  - `data/existing_scrna_diversity.csv` — 从 27 行扩到 ≥60 行，字段: `abs_path, filename, GSE, tissue, organism, cell_count, provenance_folder, modality=scrna`
  - `data/existing_scatac_inventory.csv` — 新文件，同字段 + `modality=scatac`
  - `scripts/inventory_host_h5ads.py` — 可重入脚本（读每个 h5ad，取 `n_obs`、obs/uns 中的 tissue/organism、从文件名正则出 GSE）
- **依赖**: 无（可与 T1 并行）
- **验收**: AC-8
- **做法**: 用 `anndata.read_h5ad(..., backed='r')` 快速读 shape + obs 列名（避免全量加载）；GSE 正则 `GSE\d+|GSM\d+`；tissue 优先 obs 元数据，失败则从文件名启发。

### T3. scATAC 源定位 + drop manifest 填充
- **Tier**: haiku （机械：比对 `archived_extra_scATAC/` 的 CSV 文件名 vs `~/Downloads/ATAC_data/` 中的 h5ad）
- **输入**:
  - `/home/zeyufu/LAB/CCVGAE/CG_results/archived_extra_scATAC/` （115 CSV 文件名）
  - `~/Downloads/ATAC_data/` （≥30 h5ad，已确认含 `GSE198730_*`, `GSM5402771_*`, `GSM5766892_*`, `GSM5769453_*` 等对应 `dropped_scatac_v2.csv` 的 GSE/GSM）
  - `data/dropped_scatac_v2.csv` （15 行 cell_count 空）
- **输出**:
  - `scripts/setup_symlinks.sh` — 把 `~/Downloads/ATAC_data` 加入 `SCRNA_SOURCES`（或建并列的 `SCATAC_SOURCES` 数组，更清晰）
  - `data/dropped_scatac_v2.csv` — 15 行 `cell_count` 填完
  - `data/existing_scatac_inventory.csv` — 由 T2 扩展，并确认 `h5ad scATAC: N>0`
- **依赖**: T2（T2 已扫完 ATAC_data）
- **验收**: AC-4 + AC-5 (setup_symlinks 部分)
- **做法**: 从 `archived_extra_scATAC/*.csv` 解析 filename key → 去掉 `CG_*_series_`、`_dfs.csv` → 匹配 `~/Downloads/ATAC_data/*.h5ad`；对命中者用 anndata backed 读 `n_obs` 写入 drop manifest；把 `ATAC_data/` 加进 setup_symlinks 并 re-run，断言 `n_scatac_h5ad > 0`。

### T4. Bug #1 — metrics binning `duplicates="drop"`
- **Tier**: sonnet
- **输入**: `scccvgben/training/metrics.py`（已确认 27 列 schema），`tests/test_smoke_pipelines.py`
- **输出**:
  - `scccvgben/training/metrics.py` — 所有 `pd.cut` / `pd.qcut` 调用补 `duplicates="drop"`（当前文件未见直接 cut 调用，需 grep 确认真实 binning 位点；极可能在 `_intrinsic_metrics` 的 trajectory_directionality/noise_resilience TODO 未来实现里，或在 UMAP/tSNE Q-metric 代码路径的外部导入处。先 **grep 全包** `\bqcut\b|\bcut\(` 以锁定真实位置；若目前无 cut 调用，该 bug 的实际触发点在 Q_local/Q_global/K_max 的 UMAP/tSNE 路径 — 当前为 NaN TODO，需在补这些指标时一并加 `duplicates="drop"`）
  - `tests/test_metrics_binning_regression.py` — 输入：退化 latent（全 0 / 全同值 / 2 个唯一值）；断言：产出 DataFrame 行所有 non-NaN 列都是 `np.isfinite`，且当列不是 NaN 时不抛 `ValueError: Bin edges must be unique`
- **依赖**: 无
- **验收**: AC-2
- **做法**: (1) `rg -n "qcut|pd\.cut" scccvgben/` 定位真实 binning（可能在 `graphs/construction.py` 的 neighbor 量化、或 metrics 的 quality_umap 计算尚待实现处）；(2) 给每处加 `duplicates="drop"`；(3) 补回归测试；(4) 运行 `scripts/run_encoder_sweep.py` 对 `workspace/data/scrna/adata.h5ad` 或已有真 h5ad 做 1 数据集 smoke — CSV 中 ASW/DAV/CAL 列必须数值化。

### T5. Bug #2 — GATv2 silent NaN
- **Tier**: sonnet
- **输入**: `scccvgben/models/encoder_registry.py:37-42`, `scccvgben/models/ccvgae.py`, `tests/test_encoder_registry.py`, `results/encoder_sweep/synthetic_200x100.csv`
- **输出**:
  - `scccvgben/models/encoder_registry.py` — 修复 GATv2 init（最可能根因：`heads=4, concat=False, dropout=0.1` 与调用侧传入的 `in_channels/out_channels` 组合导致 `out_dim` 被除以 head；或 PyG 版本中 `GATv2Conv` 需要 `add_self_loops` 显式 True；或 forward 时未传 edge_weight）
  - `tests/test_encoder_registry.py` — 增 `test_gatv2_produces_finite_output`：合成 `x=(200,100) ~ N(0,1)`，kNN-Euc 图，跑 GATv2 一步 forward，断言 `torch.isfinite(out).all()`
- **依赖**: 无（可与 T4 并行）
- **验收**: AC-3
- **做法**: (1) 单独构造 200×100 合成输入，pytest -k gatv2 重现 NaN；(2) 打印 `out.std()`, `out.mean()`, 查 edge_weight dtype / NaN / inf；(3) 常见 fix：`fill_value=0.0` for edge_attr、`add_self_loops=True`；(4) 重跑 `scripts/run_encoder_sweep.py --encoders GATv2 --smoke`，CSV GATv2 行全列有限。

### T6. Bug #5 — 死 import `nb_loss`
- **Tier**: haiku
- **输入**: `scccvgben/training/trainer.py:9`
- **输出**: `trainer.py` — import 行去掉 `nb_loss`（保留 `mse_loss, kl_loss, adj_loss`）；如 `losses.py` 中 `nb_loss` 函数仍被外部引用（grep 确认无引用）可保留定义但不 import。
- **依赖**: 无
- **验收**: AC-5 (dead import 子项)
- **做法**: `rg -n "nb_loss" scccvgben/ scripts/ tests/` → 确认只剩 losses.py 的定义；编辑 trainer.py 第 9 行；pytest 绿即可。

### T7. Bug #6 — DRY label extraction helper
- **Tier**: sonnet
- **输入**:
  - `scccvgben/data/loader.py:60-77`（scan order: cell_type → leiden → celltype → label → labels → louvain → cluster）
  - `scccvgben/baselines/runner.py:84-89` (_get_labels: label → cell_type → celltype → leiden → louvain → cluster) **— 注意两处顺序和集合略不同**
- **输出**:
  - `scccvgben/data/labels.py` — 新模块，`extract_labels(adata, as_codes: bool = False) -> np.ndarray | None`；统一候选列顺序；`as_codes=True` 时返回 int64 codes，否则返回 str（满足 runner 当前使用 str、loader 当前转 codes 的两种消费）
  - `scccvgben/data/loader.py` — 调用 `extract_labels(..., as_codes=True)`
  - `scccvgben/baselines/runner.py` — 调用 `extract_labels(..., as_codes=False)`
  - `tests/test_label_extraction.py` — 覆盖：缺全部列 / 仅 `leiden` / 多列齐全（断言优先级）/ `label` 列为 category dtype / 空 obs
- **依赖**: 无
- **验收**: AC-5 (DRY 子项)
- **关键判断**: 两处原本顺序 **不同** — 新 helper 必须统一为一个明确顺序（建议: `label → cell_type → celltype → leiden → louvain → cluster → labels`）。在 PR/commit message 中必须记录此行为变更，因为 loader 原本先看 cell_type，而 runner 原本先看 label。**这属于可观察行为改动，Critic 需要审。**

### T8. Bug #7 — GH Actions pin to SHAs
- **Tier**: haiku
- **输入**: `.github/workflows/pages.yml`（当前用 `actions/checkout@v4` 等 4 处未 pin）
- **输出**: `pages.yml` — 4 个 `uses:` 行改为 `<action>@<40-char SHA>` + 行尾注释 `# v4.1.7` 标注语义版本
  - `actions/checkout@v4` → `actions/checkout@<SHA>` # v4.1.7
  - `actions/configure-pages@v4` → `@<SHA>` # v4.0.0
  - `actions/upload-pages-artifact@v3` → `@<SHA>` # v3.0.1
  - `actions/deploy-pages@v4` → `@<SHA>` # v4.0.5
- **依赖**: 无
- **验收**: AC-5 (SHA pin 子项)
- **做法**: `gh api repos/actions/checkout/git/ref/tags/v4.1.7 --jq .object.sha`（需网络）→ 替换；或用 OpenSSF 推荐的 `pinact` 工具。**必须验证 SHA**，手打不可。

### T9. Bug #8 — setup_symlinks.sh 路径参数化
- **Tier**: haiku
- **输入**: `scripts/setup_symlinks.sh:19-20` (hardcoded `/home/zeyufu/LAB/scCCVGBen`, `/home/zeyufu/LAB/CCVGAE`)
- **输出**: `setup_symlinks.sh` —
  - `REPO="${REPO:-$(git rev-parse --show-toplevel 2>/dev/null)}"`
  - `SRC="${SRC:-${REPO%/*}/CCVGAE}"` （或保留当前 default 但允许 env 覆盖）
  - `SCRNA_SOURCES` 数组默认保留绝对路径但允许 `SCRNA_SOURCES_EXTRA` env 追加
  - 所有 `/home/zeyufu/Downloads` 引用替换为 `${HOME}/Downloads`
  - header 加 usage 示例说明 env overrides
- **依赖**: T3（T3 会向 SCRNA_SOURCES/SCATAC_SOURCES 追加 ATAC_data）— 建议先 T3 再 T9，避免冲突。**反向也可：先 T9 重构后 T3 用新接口追加，更干净。推荐此顺序。**
- **验收**: AC-5 (parameterization 子项)
- **做法**: 重跑 `bash scripts/setup_symlinks.sh` 验证无 warning、counts 正确；另用 `REPO=/tmp/fake bash scripts/setup_symlinks.sh 2>&1` 验证 env 覆盖生效。

### T10. Bug #9 / AC-6 — `fetch_geo_scrna.py` 真实下载器
- **Tier**: opus （网络设计 + 协议选择 + 鲁棒性 + 幂等 + 指标过滤）
- **输入**:
  - 当前 skeleton `scripts/fetch_geo_scrna.py`（第 78 行 raise NotImplementedError）
  - `pyproject.toml`（需加依赖）
  - 已确认 **GEOparse 未安装**，requests 已装
- **输出**:
  - `pyproject.toml` — dependencies 加 `geoparse>=2.0.3` 和 `h5py`；保持 `requires-python>=3.10`
  - `scripts/fetch_geo_scrna.py` — 真实实现，关键设计点：
    - **协议**：优先 GEOparse `get_GEO(geo=GSE, destdir=...)`；若 `GEOparse` 抛 URLError → fallback 到 NCBI E-utilities via `requests`（HTTPS to `https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{GSE}/suppl/`）。**不走 rsync / 裸 FTP**（spec 约束）。
    - **目标文件类型**：只下 `*filtered_feature_bc_matrix.h5` 或 `*_raw_counts.h5ad`，并在下载后用 `h5py` 探头确认是 raw count（`data` 内应为 int32/uint16）+ `n_cells * n_genes < 100M` (reject atlas-scale; HCL≈1.4M cells)。
    - **h5 → h5ad 转换**：用 `scanpy.read_10x_h5`；保留 obs/var；attach `GSE`, `tissue`, `organism`, `source_url` 到 `.uns['metadata']`。
    - **幂等**：output 路径 `workspace/data/scrna/{GSE}.h5ad` 存在且 size>0 → skip。
    - **日志**：`workspace/logs/fetch_geo.log`（file handler，FORMAT 含 timestamp / GSE / bytes / duration / outcome）。
    - **CLI**：保留 `--target`, `--candidate-pool`, `--out`, `--candidate-csv`；新增 `--proxy-env`（默认 read from `HTTPS_PROXY`）和 `--max-retries=3`。
    - **失败处理**：accumulate `failures: list[dict{gse, reason, http_status}]` → 写 `workspace/logs/fetch_failures.csv`（非 .txt，以便 handoff 解析）。
    - **限速**：每次下载后 `time.sleep(2.0)` + exponential backoff on HTTP 429/503。
  - `tests/test_fetch_geo_dry_run.py` — unittest.mock stub GEOparse.get_GEO，断言：(a) dry-run 不发网络请求；(b) 已存在文件触发 skip；(c) candidate pool 限制正确；(d) 失败 GSE 写入 failures.csv
- **依赖**: 无（但真实下载在 T13，需 T10 完成后才能执行）
- **验收**: AC-6
- **关键判断**: **GEOparse 加依赖会影响 pyproject 与 CI**。需 Critic 审视是否要加 optional extra `geo = ["geoparse"]` 而非核心 dep，以减小 baseline footprint。推荐加到 optional extra，downloader 在 import 时 try/except 提示用户 `pip install scccvgben[geo]`。

### T11. Candidate pool 扩张（≥60 GSE + curation script）
- **Tier**: sonnet
- **输入**:
  - `data/scrna_candidate_pool.csv`（当前 10 行）
  - `/home/zeyufu/LAB/CCVGAE/.omc/specs/deep-interview-scccvgben-v2.md`（上游 diversity rubric）
  - T1 产物 `revised_methodology_manifest.csv`
  - T2 产物 `existing_scrna_diversity.csv`
- **输出**:
  - `scripts/curate_candidate_pool.py` — 可重入；算法：(1) 从 upstream spec 读 rubric（tissue/organism/disease 分布目标）；(2) 对 existing inventory 做 histogram；(3) 用 NCBI E-utilities `esearch.fcgi?db=gds&term=...` 查找 under-represented 类别的 GSE；(4) 按 diversity gain 排序；(5) 输出 `data/scrna_candidate_pool.csv` ≥ 60 行，带 `priority, rationale, estimated_cells` 列。
  - `data/scrna_candidate_pool.csv` — 扩到 ≥60 行
  - `data/diversity_rubric.md` — rubric 规则人读版（用于 Critic 审）
- **依赖**: T1 (revised manifest), T2 (inventory)
- **验收**: AC-9
- **做法**: 优先从上游 spec 既有 candidate（若有）引入；E-utilities 查询用 `requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi', params={...})`；diversity 评分用 `1 / (1 + count_in_existing[tissue])`。

### T12. Reconcile + residual gap 计算
- **Tier**: sonnet
- **输入**: T1 manifest, T2 inventory, T11 candidate pool
- **输出**:
  - `scripts/compute_residual_gap.py` — 生成 `data/residual_gap_report.md`：当前有 X 个 scRNA / Y 个 scATAC on-host，revised 需要 N_scrna + N_scatac，缺口 `gap_scrna = max(N_scrna - X, 0)`
  - `data/residual_gap_report.md` — 报表 + 下载目标数字（T13 --target 值）
- **依赖**: T1, T2, T11
- **验收**: AC-10 (part of)
- **做法**: 简单集合运算：`need_set - have_set`；若 have_set > need_set，取 diversity 最高 N 个。

### T13. 执行真实 GEO 下载
- **Tier**: sonnet （执行 + 监控；不需 opus）
- **输入**: T10 的 downloader, T11 candidate pool, T12 residual gap
- **输出**:
  - `workspace/data/scrna/<GSE>.h5ad` × 实际 gap 数量
  - `workspace/logs/fetch_geo.log`
  - `workspace/logs/fetch_failures.csv`
- **依赖**: T10 ✓, T11 ✓, T12 ✓
- **验收**: AC-6 (execution part), AC-10 (execution part)
- **做法**:
  ```bash
  python scripts/fetch_geo_scrna.py \
      --target $(cat data/residual_gap_report.md | grep gap_scrna | awk '{print $2}') \
      --candidate-pool 60 \
      --out workspace/data/scrna/ \
      --candidate-csv data/scrna_candidate_pool.csv
  ```
  建议 `run_in_background=True`（可能跑数小时）。**限速导致真实下载数 < target 不是致命失败**：在 HANDOFF 里记录失败 GSE 队列让下 session 重试即可。

### T14. Rebuild `datasets.csv`
- **Tier**: sonnet
- **输入**: `scripts/build_datasets_csv.py`（已有），`workspace/data/{scrna,scatac}/`
- **输出**: `scccvgben/data/datasets.csv` — full 200-row（或 revised count）
- **依赖**: T13 (scrna 下载完), T3 (scatac 源落位)
- **验收**: AC-10 (part of)
- **做法**: `python scripts/build_datasets_csv.py --build-canonical`；如原脚本 bug 需 patch。

### T15. Smoke + verify
- **Tier**: sonnet
- **输入**: T14 产物 + T4/T5 fix
- **输出**:
  - `scripts/verify_benchmark_size.py` 输出（stdout + exit code 0）
  - `results/encoder_sweep/smoke_real.csv` — 单一 scRNA + 单一 scATAC，跑 GATv2 + GAT 一圈，所有列数值化
- **依赖**: T4, T5, T14
- **验收**: AC-1, AC-2, AC-3, AC-10
- **做法**:
  1. `pytest tests/ -q` 全绿（31 + 新增 ≥ 3 regression 测试）
  2. `python scripts/verify_benchmark_size.py` → 打印 100 + 100 = 200（或 revised）
  3. `python scripts/run_encoder_sweep.py --encoders GAT,GATv2 --datasets <1_scrna_key>,<1_scatac_key> --smoke --out results/encoder_sweep/smoke_real.csv`
  4. grep NaN smoke_real.csv — 所有预期有值的列必须是数字

### T16. HANDOFF_GPU_SESSION.md
- **Tier**: sonnet
- **输入**: 所有上游 task 的输出清单
- **输出**: `HANDOFF_GPU_SESSION.md`（repo root）必须包含：
  1. **Files changed** — git log 汇总 + 每 commit 一行描述
  2. **New scripts** — `inventory_host_h5ads.py`, `curate_candidate_pool.py`, `compute_residual_gap.py`
  3. **Assumptions log** — 每个有歧义决策一行（如 DRY helper 的新优先级顺序、是否加 geoparse 到 optional extra、是否保持 200）
  4. **GSE queue** — 成功/失败 列表（来自 fetch_failures.csv）
  5. **STAGE 2 launch checklist** — 精确复刻 `LAUNCH_BENCHMARK.md` STAGE 2 的三个命令，并注明 GPU preflight (`nvidia-smi`)、pre-commit 已装、`workspace/` 可写
  6. **Known residuals** — 本 session 未解的问题（如下载未达 target）
- **依赖**: T1–T15
- **验收**: AC-11

### T17. Focused commits + pre-commit hook
- **Tier**: haiku（机械 commit 分组）
- **输入**: 所有上游 diffs
- **输出**: git log 新增 8–10 commits：
  1. `fix(trainer): remove dead nb_loss import (#5)` — T6
  2. `fix(metrics): duplicates='drop' in binning (#1) + regression test` — T4
  3. `fix(encoder): GATv2 init + finite-output test (#2)` — T5
  4. `refactor(data): extract label helper for loader+runner (#6)` — T7
  5. `ci(pages): pin GH Actions to SHAs (#7)` — T8
  6. `chore(setup_symlinks): parameterize paths + add SCATAC_SOURCES (#8, #3)` — T3+T9
  7. `feat(inventory): scan host h5ads → existing_scrna/scatac_inventory.csv` — T2
  8. `feat(curate): expand candidate pool to 60+ with diversity rubric` — T11
  9. `feat(fetch): real GEO downloader via GEOparse + proxy fallback (#9)` — T10
  10. `feat(data): executed GEO fetch + rebuild datasets.csv to 200` — T13+T14
  11. `docs(handoff): HANDOFF_GPU_SESSION.md + smoke verification` — T15+T16
- **依赖**: 每个 commit 前 `pytest -q` 绿
- **验收**: AC-12, AC-1 (pytest 绿各 commit)
- **做法**: 用 `git add <specific_files>` 避免误提交 `workspace/`；每 commit 后跑 `pytest tests/ -q`，失败就 amend 或新 commit 修。

---

## 3. 并行化策略

### 可并行组 A（"Bug fixes"，全部零依赖）
- T4 (Bug #1 metrics binning)
- T5 (Bug #2 GATv2 NaN)
- T6 (Bug #5 dead import)
- T7 (Bug #6 DRY helper)
- T8 (Bug #7 GH SHA pin)
- T9 (Bug #8 setup_symlinks 参数化)

→ 每个都可由独立 sonnet/haiku worker 并行做，共用 `pytest tests/` 会引入轻微冲突（建议每 worker 各自 `pytest -k <module>`），merge 时统一跑一次 full suite。**6 个可全并行**。

### 可并行组 B（"Inventory"，只依赖外部读）
- T1 (read notebooks)
- T2 (scan h5ads)

→ 完全独立，opus + sonnet 同时跑。

### 串行依赖链 C（"Data pipeline"）
```
(T2 done) ─> T3 (scATAC source)
(T1 + T2 done) ─> T11 (candidate pool) ─> T12 (residual gap) ─> T13 (download) ─> T14 (datasets.csv)
```
T13 是**外部瓶颈**（下载时长），此期间 worker 可去做 T4–T9 的补漏或预写 T16 骨架。

### 收尾串行 D
```
T3 + T9 merged
T4 + T5 merged
T14 done
  ─> T15 (smoke + pytest)
  ─> T16 (handoff doc)
  ─> T17 (commits)
```

### 推荐 Phase 2 worker 拓扑
- **Worker-Alpha (opus)**: T1 → T10（downloader 实现）→ T16 骨架预写
- **Worker-Beta (sonnet)**: T2 → T3 → T11 → T12 → T13 启动 (bg) → T14 → T15
- **Worker-Gamma (sonnet, bug 专线)**: T4 → T5 → T7（顺序：有 metrics/encoder/helper 的 pytest 依赖）
- **Worker-Delta (haiku, 清洁工)**: T6 → T8 → T9（全是 mechanical diff）
- **Final (任一)**: T17 commits

关键路径仍受 T13 下载时长支配。

---

## 4. 风险与应对（Top 7）

| # | 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|---|
| R1 | **GEO 代理 API 限速/429** 导致下载远低于 target | 高 | 中 | (a) downloader 内置 exp backoff + max-retries=3；(b) candidate pool 扩到 ≥60 提供缓冲；(c) 把 shortfall 作为 known residual 写 HANDOFF 而非 fail session；(d) spec AC-10 明确允许"revised count with justification"，所以**部分完成 = 完成** |
| R2 | **revised notebook 数据 manifest 解析失败** (notebook 代码散落、data path 是 local filesystem) | 中 | 高 | (a) 先跑 `jupyter nbconvert --to script` 批量出 .py；(b) 回退方案：读 `.omc/specs/` 里可能的 structured spec；(c) 极端回退：守 200 默认并在 handoff 标 "revised manifest inference limited"；(d) 该任务是 T1 = opus 级，不给 haiku |
| R3 | **scATAC 源匹配不完整**（archived CSV 115 vs ATAC_data h5ad ≤30） | 中 | 高 | (a) 先看 T2 扫完后到底有多少 scATAC on-host — 如果 < 100，说明 scATAC 100 目标需 GEO 补；(b) scATAC GEO 下载**不在本 session spec 内**（spec 只说 scRNA），所以 scATAC 缺口只能作为 known residual 到 HANDOFF；(c) 必须明确告知 user：100 scATAC 目标或将降到"on-host 可得数量"—需要 Critic 审这个默认决策 |
| R4 | **pre-commit hook 失败**（ruff/mypy 对新代码报错） | 中 | 低 | (a) T17 前先 `pre-commit run --all-files`；(b) 若 ruff/mypy 规则过严，按 spec 要求遵守而不 skip `--no-verify`；(c) 新文件 type hint 完整 |
| R5 | **GEOparse 依赖引入冲突**（与 torch / scanpy 的 h5py 版本锁） | 中 | 中 | (a) 加为 optional extra `[geo]` 而非核心 dep（减耦合）；(b) 在 sandbox venv 试装确认无冲突；(c) 若冲突，E-utilities via requests 作为 pure-Python fallback |
| R6 | **binning bug 实际位点找不到**（metrics.py 当前不含 `cut`/`qcut` 调用） | 中 | 中 | (a) grep 全仓 `rg "qcut\|pd\.cut"`；(b) 若确实不在 code 中，bug 可能在外部 lib（umap-learn 的 connectivity）— 此时 fix 改为 "在 metrics wrapper 里 catch `ValueError: Bin edges must be unique` 并填 NaN" + 记录在 KNOWN_ISSUES 作为"已缓解但非根因修复"；(c) 该决策需 user 或 Critic 签字 |
| R7 | **label helper 行为变更引入回归**（两处原本顺序不同） | 中 | 中 | (a) 新统一顺序需在 commit message 明确声明；(b) 跑所有下游脚本 on 至少 3 个真实 h5ad，对比前后 labels；(c) Critic 必审 |

---

## 5. Phase 2 执行顺序建议

### Batch 1（并行，~2 h）
- T1（notebooks 解析，opus bg）
- T2（inventory 扫盘，sonnet bg）
- T6 + T8（haiku，5 min 搞定）
- T9（setup_symlinks 参数化，准备好 T3 要追加的接口）

### Batch 2（并行，~2 h；由 Batch 1 的结果 unlock）
- T3（scATAC 源 + drop manifest，T2 done 后）
- T4 + T5 + T7（bug fixes，zero deps，可独立并行）
- T10（downloader 实现，opus）
- **merge + pytest 全绿**

### Batch 3（串行 + 一个长 bg，~4–10 h）
- T11（candidate pool 扩张，sonnet）
- T12（residual gap 计算，sonnet）
- T13（**启动真实下载，bg**，长 job）
- **等 T13 期间**：工作者可去起草 T16 HANDOFF 骨架、完善 test coverage、补 docstring

### Batch 4（收尾，~1–2 h；等 T13 完成）
- T14（rebuild datasets.csv）
- T15（smoke + verify_benchmark_size）
- T16（HANDOFF 终稿）
- T17（分组 commits + pytest 每个 commit 绿）

### 如果时间超支（超过 16 人时）
- 降低 T13 target（spec 允许）
- 把 T4 改为 "in-wrapper catch + TODO note"（R6 兜底）
- T11 的 60 个 candidate 如 E-utilities 查询耗时太长，降到 ≥ 60 中 "≥ 45 新 + 15 来自上游"
- **不可降**: 9 bug 全修（硬约束）、pytest 绿（硬约束）、handoff doc（AC-11）

---

## 6. Phase 3 QA 策略

### 6.1 Unit tests（pytest，每 commit 跑）
- `tests/test_metrics_binning_regression.py` — T4 产出
- `tests/test_encoder_registry.py::test_gatv2_produces_finite_output` — T5 扩展
- `tests/test_label_extraction.py` — T7 产出（5 个 case：缺全、仅 leiden、全齐、empty obs、category dtype）
- `tests/test_fetch_geo_dry_run.py` — T10 产出（mock GEOparse，不发网络）
- `tests/test_inventory_scan.py` — T2 产出（fixture 小 h5ad 目录，断言 CSV schema 正确）

覆盖 AC-1, AC-2, AC-3, AC-5 (DRY), AC-6 (idempotency)。

### 6.2 Integration smoke（pytest -m integration，每 batch 结束跑）
- `tests/test_smoke_pipelines.py::test_real_h5ad_encoder_sweep` — 扩展现有 smoke 用 real h5ad （如 `~/Downloads/DevelopmentDatasets/adata.h5ad`），跑 GAT + GATv2 一圈，断言 CSV 无 NaN in non-TODO 列
- `tests/test_symlinks.py::test_scatac_nonzero` — T3 产出后，断言 `setup_symlinks.sh` 输出 `n_scatac_h5ad > 0`

覆盖 AC-1, AC-2, AC-3, AC-4。

### 6.3 End-to-end 手动 verify（人跑，T15 内）
1. Fresh clone → `pip install -e ".[dev,geo]"` → `pytest tests/ -q` 应全绿
2. `bash scripts/setup_symlinks.sh` → 报 n_scrna_h5ad > 0 且 n_scatac_h5ad > 0
3. `python scripts/fetch_geo_scrna.py --target 1 --candidate-csv data/scrna_candidate_pool.csv --dry-run` → 正常列出
4. `python scripts/verify_benchmark_size.py` → 打印 200（或 revised）
5. 单数据集 smoke encoder sweep（见 T15）→ CSV 所有非 TODO 列数值化

覆盖 AC-6, AC-10, AC-11。

### 6.4 Commit-gated 检查（T17）
- `pre-commit run --all-files` 绿
- `pytest tests/ -q` 绿
- `gh workflow list` 确认 pages.yml 仍 parse OK

---

## 7. Phase 4 Validation 侧重点

| AC | 审查类型 | Owner | 重点 |
|----|----|----|----|
| AC-1 pytest ≥31/31 | **代码审** | executor 自查 | 机械验证 |
| AC-2 binning fix | **代码审 + 架构审** | Critic | R6 决策（若根因在外部 lib 是否接受 wrapper 兜底）需架构审；新回归测试覆盖是否充分需代码审 |
| AC-3 GATv2 finite | **代码审** | executor | smoke CSV 展示即可 |
| AC-4 scATAC 源 | **架构审** | Critic | 如 scATAC on-host < 100，是否接受降目标需架构审 |
| AC-5 cleanup 四子项 | **代码审** | executor | T7 label helper 行为变更 **需 Critic 额外审**（统一优先级顺序） |
| AC-6 GEO downloader | **代码审 + 安全审** | Critic + 安全审 | 安全审：是否泄漏 proxy credential 到 log；是否 follow redirect 到可信 host；h5 文件 size/magic 校验是否充分（防路径遍历、防 zip-bomb 式 h5） |
| AC-7 revised manifest | **架构审** | Critic | revised methodology 是否真指向 200 数据集；notebook 解析是否完备 |
| AC-8 inventory CSV | **代码审** | executor | schema 一致性（路径字段必须绝对） |
| AC-9 candidate pool ≥60 | **架构审** | Critic | diversity rubric 是否忠于上游 spec；curation 脚本可重入 |
| AC-10 真实下载 | **代码审 + 架构审** | Critic | shortfall 的处理策略（known residual 到 handoff）是否被 user 认可 |
| AC-11 HANDOFF doc | **架构审** | Critic | 是否真的能让下 session 无沟通启动；checklist 的可操作性 |
| AC-12 focused commits | **代码审** | executor | 每 commit 是否 atomic；无 workspace/ 误提交；无 credential 泄漏 |

### 必须 Critic 重点审的设计决策
1. **DRY label helper 的统一顺序**（T7）— 改了两处消费者原本不同的优先级，属行为变更。
2. **GEOparse 作为 optional extra 还是核心依赖**（T10）— 影响 pyproject + CI matrix。
3. **Bug #1 如果根因在外部 lib 是否用 wrapper 兜底**（T4 / R6）— 影响后续"真根因"是否还需挖。
4. **scATAC on-host < 100 时的目标缩减**（T3 / R3）— 影响 paper claim。
5. **revised methodology 是否真支持 200 总数**（T1）— 影响整个 session 的数字目标。
6. **T13 真实下载 shortfall 的降级到 handoff 是否可接受**（R1）— 影响 AC-10 判定。
7. **setup_symlinks.sh 是新增 SCATAC_SOURCES 数组还是并入 SCRNA_SOURCES**（T3+T9）— 可读性 vs 最小变更。

---

## 8. 文件产出清单（Phase 2 executor 需交付）

### 新增
- `scripts/inventory_host_h5ads.py` (T2)
- `scripts/curate_candidate_pool.py` (T11)
- `scripts/compute_residual_gap.py` (T12)
- `scccvgben/data/labels.py` (T7)
- `data/existing_scatac_inventory.csv` (T2)
- `data/revised_methodology_manifest.csv` (T1)
- `data/residual_gap_report.md` (T12)
- `data/diversity_rubric.md` (T11)
- `tests/test_metrics_binning_regression.py` (T4)
- `tests/test_label_extraction.py` (T7)
- `tests/test_fetch_geo_dry_run.py` (T10)
- `tests/test_inventory_scan.py` (T2)
- `HANDOFF_GPU_SESSION.md` (T16)
- `.omc/plans/revised-methodology-notes.md` (T1)

### 修改
- `scripts/fetch_geo_scrna.py` (T10) — skeleton → 真实
- `scripts/setup_symlinks.sh` (T3+T9) — 参数化 + SCATAC 源
- `scccvgben/training/trainer.py` (T6) — 去 nb_loss import
- `scccvgben/training/metrics.py` (T4) — duplicates='drop'
- `scccvgben/models/encoder_registry.py` (T5) — GATv2 kwargs
- `scccvgben/data/loader.py` (T7) — 用新 helper
- `scccvgben/baselines/runner.py` (T7) — 用新 helper
- `scccvgben/data/__init__.py` (T7) — export extract_labels
- `.github/workflows/pages.yml` (T8) — SHA pin
- `pyproject.toml` (T10) — 加 geoparse（optional extra 推荐）
- `data/existing_scrna_diversity.csv` (T2) — 27 → ≥60
- `data/scrna_candidate_pool.csv` (T11) — 10 → ≥60
- `data/dropped_scatac_v2.csv` (T3) — 15 行 cell_count 填充
- `scccvgben/data/datasets.csv` (T14) — 重建到 200
- `tests/test_encoder_registry.py` (T5) — 新 finite 测试
- `tests/test_smoke_pipelines.py` (T15) — 真实 h5ad case

### Workspace（gitignored，但 commit 其 metadata）
- `workspace/data/scrna/*.h5ad` (T13) — **新下载文件**
- `workspace/logs/fetch_geo.log` (T13)
- `workspace/logs/fetch_failures.csv` (T13)
- `results/encoder_sweep/smoke_real.csv` (T15)

---

## 9. 退出条件（Phase 2 结束时必须成立）

1. `pytest tests/ -q` 绿，count ≥ 35（原 31 + 至少 4 个新回归测试）
2. `bash scripts/setup_symlinks.sh` 报 `h5ad scRNA: N1>0, scATAC: N2>0`
3. `python scripts/verify_benchmark_size.py` 打印目标数（200 或 revised）且 exit 0
4. `python scripts/fetch_geo_scrna.py --help` 显示完整 CLI（无 NotImplementedError）
5. `grep -rn "NotImplementedError" scripts/ scccvgben/` 无 hit
6. `grep -n "nb_loss" scccvgben/training/trainer.py` 无 hit
7. `cat .github/workflows/pages.yml | grep -c "@v[0-9]"` = 0（全部 SHA pinned）
8. `ls HANDOFF_GPU_SESSION.md` 存在且包含 STAGE 2 checklist section
9. `git log --oneline main..HEAD | wc -l` ≥ 8（分组 commits）
10. `git status` clean（无 untracked 要 commit 的）

---

## 10. 给 Phase 2 executor 的第一条指令建议

```
启动：并行起 T1 (opus) + T2 (sonnet) + T6 (haiku) + T8 (haiku)。
等 T2 完成后立刻起 T3。
等 T1 完成后评估 revised manifest 是否动摇 200 数字 —— 如动摇，
立刻升级到 ralplan consensus 让 user 签字再继续。
```
