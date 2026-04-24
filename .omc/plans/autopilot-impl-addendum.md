# autopilot-impl.md — Decision Addendum (2026-04-23)

本 addendum 记录 user 在 Phase 1 Critic 审议后对 6 个决策点的最终签字。
autopilot-impl.md 主文件 + 本 addendum 共同构成 Phase 2 的执行依据。
冲突时 **addendum 优先**。

---

## D1 — AC-2 / T4 UMAP/tSNE Q 指标的交付方式

**决策**：**端口迁移**，不要新实现。

**源代码位置**：
- `/home/zeyufu/LAB/CCVGAE/DRE.py` 的 `evaluate_dimensionality_reduction(X_high, X_low, k=10)`
  - 返回 dict：`distance_correlation, Q_local, Q_global, K_max, overall_quality`
  - 核心：`compute_qnx_series` (L157-191), `get_q_local_global` (L193-231),
    `get_coranking_matrix` (L120-153), `get_ranking_matrix` (L84-116)
- `/home/zeyufu/LAB/CCVGAE/LSE.py` 的 `trajectory_directionality_score()`
  - PCA 主成分支配比计算轨迹方向性
- 依赖：`scipy.stats.spearmanr`, `sklearn.metrics.pairwise_distances`,
  `sklearn.decomposition.PCA`

**执行**：
1. 把 `DRE.py` 的相关函数拷贝（或整体 include）到 `scccvgben/training/metrics.py` 的合适
   辅助模块，例如 `scccvgben/training/dre.py` 新建一个文件。
2. 在 metrics.py 的 UMAP/tSNE 段把 `np.nan` 替换为真实计算。
3. AC-2 验收升级为：real-h5ad smoke run 的 `*_umap` 和 `*_tsne` 列都必须是 `np.isfinite`。
4. `trajectory_directionality_intrin` 同样从 LSE.py 端口。

**AC-2 判定**：端口完成 + real-h5ad smoke run 所有 metric 列都 `np.isfinite` → AC-2 PASS。

---

## D2 — T7 Label 统一策略（不是 fallback chain，而是 KMeans 先验）

**决策**：**不要**保留 loader.py/runner.py 的 label fallback chain。**统一改为**：
- 使用 CCVGAE 已验证的 `get_labels()` + `compute_metrics()` 组合
- 本质上是 "KMeans 先验 + latent-space KMeans"（对"标签保留效果"的量化）

**源代码位置**：
- `/home/zeyufu/LAB/CCVGAE/CCVGAE_supplement/run_hyperparam_sensitivity.py`
  - `get_labels(adata)`: fallback chain
    `cell_type → celltype → label → labels → cluster → clusters → annotation → CellType`
    → 最后尝试任何 category/nunique<100 列
    → 若全无返回 `None`
  - `compute_metrics(latent, labels, adata)`:
    * `n_clusters = len(np.unique(labels))` 如果有真值；否则 `10`
    * `KMeans(n_clusters=n_clusters, n_init=10, random_state=42)`
    * `ref = labels if labels is not None else pred`（关键：无真值时自参照）
    * 计算 NMI, ARI

**执行**：
1. 新建 `scccvgben/data/labels.py`，端口 `get_labels()`（保留 CCVGAE 的 fallback 优先级）。
2. 把 loader.py:60-77 和 baselines/runner.py 的 `_get_labels` 都改为调用
   `scccvgben.data.labels.get_labels(adata)`。
3. metrics.py 的 `_clustering_metrics(Z, labels, n_clusters)` 已经部分实现，
   按 compute_metrics 的 `ref = labels if labels is not None else pred` 逻辑补齐。
4. T6 DRY 目标通过 D2 一次达成（#6 去重 + CCVGAE 一致性）。

**AC-5 判定**：两处调用统一到 `scccvgben.data.labels.get_labels` + 回归测试通过。

---

## D3 — scATAC 缺口策略

**决策**：**数据充足，不需任何 scATAC 下载**。

**事实基础**（user 在上一发表版本中核验）：
- `~/Downloads/ATAC_data/` 含 **115 `.h5ad`** + 38 额外 `.h5`（RAW 目录）
- 去除 cell_count 最小的 15 个 → 得到 100 scATAC（spec 目标）

**执行**：
1. T2 inventory 包含 scATAC 扫描（`*.h5ad`），schema 包含 `cell_count`。
2. T3 修 setup_symlinks.sh：把 `~/Downloads/ATAC_data/` 加入 scATAC 源
   （推荐并列建 `SCATAC_SOURCES` 数组，清晰）。
3. 扫完后按 `cell_count` 升序取 15 个写入 `data/dropped_scatac_v2.csv`
   （覆盖现有 `cell_count=blank` 的 15 行）。
4. 剩余 100 个 scATAC → canonical datasets.csv。

**AC-4 判定**：`setup_symlinks.sh` 报告 `h5ad scATAC: >=100`，`dropped_scatac_v2.csv`
的 15 行 `cell_count` 全部填齐。

**R3 完全消除**：on-host scATAC < 100 的情况 user 已确认不会发生。Non-Goals 无冲突。

---

## D4 — T17/T18 编号矛盾

**决策**：**T18 是笔误**，autopilot-impl.md 主文件的 T18 引用全部删除。
- §1.3 表格 "T17–T18" → "T17"
- §1.4 关键路径图 "T18 commits" → 直接终止于 T17
- §9 Exit 条件 #9 `≥ 8 commits` → 改为 `≥ 10 commits`（与 T17 列表的 11 项对齐，允许合并）

---

## D5 — GEOparse 依赖声明

**决策**：**optional extra `[geo]`**，不是核心依赖。

**执行**：
1. `pyproject.toml` 的 `[project.optional-dependencies]` 段加：
   ```
   geo = ["GEOparse>=2.0.3"]  # 注意：PyPI 包名是 GEOparse（驼峰）
   ```
2. 安装：`pip install -e .[geo]`
3. 主 python 文件的 `import` 必须是 `import GEOparse`（驼峰）
4. 如 `pip install GEOparse` 失败（lxml/h5py 冲突），fallback 到 E-utilities-only
   路径，代码里 `try: import GEOparse except ImportError: USE_EUTILS = True`。
5. handoff 记录实际使用的是哪一条路径。

---

## D6 — T13 真实下载 shortfall 验收

**决策**：下载 **≥ 50% target 视同 AC-10 通过**，剩余记录在 HANDOFF 作 P2 续接项。

**执行**：
1. T13 background 执行到 `fetch_geo_scrna.py --target N` 终止（内建超时或正常退出）
2. 统计成功数 S：
   - S ≥ 0.5 × N：AC-10 PASS，HANDOFF 标记 `scRNA_download_residual = N-S`
   - S < 0.5 × N：触发 ralplan consensus 请求重试 budget，或 user 仲裁

---

## User directive on engineering detail

> "首先要确保大方向正确以及终极效果，最后再同一工程化，不要让过度工程化干扰你的注意力"

**操作含义**：
1. Batch 1 (inventory + T1/T2) 完成后立即开始 Batch 2 bug 修复，**不**等全部工程细节
   定稿。
2. 遇到小决策点（setup_symlinks 路径变量命名、candidate pool CSV 新列是否预填值等），
   **executor 自决 + 在 commit message 或 HANDOFF 记录**，不上 ralplan、不问 user。
3. 只有触发**硬约束违反**（违反 spec Non-Goals / 打破 12 AC）或**外部阻塞**（网络 / 权限）
   才暂停并请 user 裁决。
4. Phase 4 Validation 关注"大方向 + 终极效果"，工程细节 note-only 不 reject。

---

## Critic 的其他 Approval Conditions 处理

以下 Critic 列的 approval condition 采用 "executor 自决 + handoff 记录" 处理（不升级）：

- ✅ T10 包名拼写：addendum D5 已明确
- ✅ T3+T9 并行冲突：合并到同一 worker（Worker-Delta: T6→T8→T9→T3）
- ✅ T2 `_RAW` 目录扫描：inventory 脚本同时扫 `*.h5` + `_RAW` 子目录
- ✅ T11 candidate_pool schema：superset `gse, description, priority, tissue,
     organism, estimated_cells`（加列不删列）
- ✅ T12 residual_gap 已扣除 workspace/ 既有 46 个 symlink
- ✅ `workspace/logs/` gitignore 保留，`data/residual_gap_target.txt` 单文件 commit

剩余 minor/open questions 交由 executor 判断，不额外决策。

---

*End of addendum. Phase 2 可立即开始。*
