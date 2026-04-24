# T1: Revised CCVGAE Methodology Summary

Generated: 2026-04-23  
Source files: CCVGAE1_SD.ipynb, CCVGAE2_UCB.ipynb, CCVGAE3_IR.ipynb,  
              CCVGAE_supplement/CGVAE1_supplement.ipynb,  
              CCVGAE_supplement/dataset_metadata_comprehensive.csv,  
              CCVGAE_snLaTeX_diff/sn-article-diff.tex,  
              CCVGAE_supplement/reviewer_comments.md

---

## 1. 数据集总数：170（非 200）

**revised CCVGAE 最终使用 170 个 benchmark 数据集：55 scRNA-seq + 115 scATAC-seq。**

scCCVGBen plan 中的默认值"200 (100+100)"与 revised CCVGAE 不符。应按 revised 实际数量：

| 模态 | 数量 | 说明 |
|------|------|------|
| scRNA-seq | 55 | 包含 2 个 revised 新增（irall GSE277292, wtko GSE278673） |
| scATAC-seq | 115 | 全部来自 Desktop/scATAC-25100/ 或 Downloads/ATAC_data/ |
| **合计** | **170** | |

额外说明：3 个额外 scRNA 数据集（GSE280145、GSE278673(UCB)、GSE280270）仅用于生物学 case study，**不计入 170 benchmark**。  
这 3 个 case study 对应三个 notebook：
- CCVGAE1_SD → `../../Desktop/CSD/1014.h5ad`（sleep deprivation, Mouse）
- CCVGAE2_UCB → `../../Desktop/UCBfiles/{CT,D4,D7,D11,D14}/`（10x MTX 格式，Human UCB 分化）
- CCVGAE3_IR → `../scRL/IRALL.h5ad`（radiation injury HSC, Mouse，irall 同时在 benchmark 中）

---

## 2. scRNA benchmark 变更（vs 初始版）

| 变更 | 详情 |
|------|------|
| 移除 | GSE120575_melanomaHmCancer（初始版有，revised 删除） |
| 移除 | GSE225948（初始版 Hematopoietic 类，revised 删除） |
| 新增 | irall (GSE277292) — Dapp1 KO LSK/HSC, Mouse, 41252 cells |
| 新增 | wtko (GSE278673) — 照射后 HSC day2, Mouse, 10224 cells |
| 净变化 | 53 kept + 2 added = 55 scRNA（增加 case-study 相关数据进 benchmark） |

---

## 3. 编码器架构（三组件 + 四变体消融）

### 核心三设计原则
1. **Centroid Inference**：推断时用确定性后验均值 μ_centroid = μ_z(x) 代替随机样本，消除采样噪声
2. **Coupling Regularization**：双重重建 bottleneck（d_c < d_z，默认 d_c=5, d_z=10），约束潜空间局部几何
3. **Graph Attention Encoder (GAT)**：k=15 NN 图（PCA/LSI 50 维空间），图注意力消息传递保留细胞邻域结构

### 模型变体（消融）
| 变体 | 说明 |
|------|------|
| VAE | 标准 VAE（MLP 编码器，推断时随机采样）—— 基线 |
| CenVAE | VAE + Centroid Inference |
| CouVAE | VAE + Coupling Regularization |
| GAT-VAE | VAE + 图注意力编码器（**注意：revised 版统一用 "GAT-VAE"，不再用 "VGEA"**） |
| CCVGAE | 全组合（三原则联合） |

**重要修订**：reviewer 指出"VGEA"命名混乱，revised 版全文统一为 **GAT-VAE**。scCCVGBen 应使用 GAT-VAE。

---

## 4. Graph 构造

- 预处理：scRNA → library-size norm + log1p + top 2000 HVG；scATAC → TF-IDF + top 2000 HV peaks
- 降维：PCA 50 维（scRNA）或 LSI 50 维（scATAC）
- KNN：k=15，Euclidean distance 在 50 维空间
- 对称化：A_ij = A_ji = 1 若任一方向为近邻
- 子图采样（subgraph_size=256）用于训练

---

## 5. Baseline 方法（revised 版明确三组）

### scRNA-seq baselines（12 个）
| 类别 | 方法 |
|------|------|
| 经典降维（7）| PCA, KPCA, ICA, FA, NMF, TSVD, DICL |
| 深度生成模型（5）| scVI, DIPVAE, InfoVAE, TCVAE, HighBetaVAE |

### scATAC-seq baselines（3 个）
| 方法 | 说明 |
|------|------|
| LSI | Latent Semantic Indexing（scATAC 标准） |
| PeakVI | SCVI-tools 峰值模态 VAE |
| PoissonVI | Poisson 似然 VAE |

---

## 6. 评估指标体系（三套 suite）

revised 版将指标明确组织为三个 axis（与 scCCVGBen Axis A/B/C 对应关系见下）：

| CCVGAE 套件 | 指标 | scCCVGBen Axis 对应 |
|------------|------|---------------------|
| BEN（clustering）| NMI, ARI, ASW, CAL, DAV, COR | **Axis A** |
| DRE（embedding quality）| DC, QL, QG, OV（UMAP & t-SNE 各4个）| **Axis B** |
| LSE（intrinsic geometry）| MD（Manifold dim）, SDR（Spectral decay）, PR（Participation ratio）, AS（Anisotropy）, TD（Trajectory directionality）, NR（Noise resilience）| **Axis C** |

**重要**：revised 版新增的指标实现细节：
- k-means 每数据集跑 10 次随机 seed，取最低 inertia 结果
- UMAP/t-SNE 固定 seed=42
- 评估全部用 μ_centroid（确定性），非随机样本
- 统计：配对 t-test（正态）或 Wilcoxon signed-rank（非正态）；多重比较用 Bonferroni 校正 + Friedman/RMANOVA 总体检验

---

## 7. 与 scCCVGBen Axis A/B/C 的匹配情况

| scCCVGBen Axis | CCVGAE 对应套件 | 状态 |
|----------------|----------------|------|
| Axis A（clustering: NMI/ARI/ASW/CAL/DAV/COR）| BEN suite | **完全匹配** |
| Axis B（embedding: DC/QL/QG/OV）| DRE suite | **完全匹配** |
| Axis C（intrinsic geometry: MD/SDR/PR/AS/TD/NR）| LSE suite | **完全匹配** |

三轴与 revised CCVGAE 完全对齐，无需调整。

---

## 8. 不该在 scCCVGBen 中做的事（revised 版特有限制）

1. **不要用 "VGEA" 命名**：revised 全文改为 GAT-VAE，scCCVGBen 应同步。
2. **不要把 case study 3 个数据集（CSD, UCBfiles, IRALL/GSE280145）纳入 benchmark**：它们仅用于生物学解释（GO-BP 富集），不计入 170 个定量评估数据集。
3. **不要期望 200 = 100+100**：revised 确认是 55+115=170，若 scCCVGBen 仍计划 200，需重新协商 scATAC 数量（需额外 +45 scATAC 或 +30 scRNA）。
4. **GO-BP 富集分析不是 benchmark 指标**：它是生物学解释步骤（Latent-gene Pearson 相关 → top-100 基因 → hypergeometric test + BH correction），不是 Axis A/B/C 的一部分。
5. **不要混淆 d_c 和 d_z**：coupling bottleneck 默认 d_c=5, d_z=10（ratio=0.5）；revised 版超参数敏感性分析已在 supplement 中完成。

---

## 9. revised 版关键性能数字（参考基准）

| 对比 | 指标 | 提升 |
|------|------|------|
| CCVGAE vs CouVAE（scRNA）| ARI, NMI | +0.104 |
| CCVGAE vs scVI（scRNA）| ASW | +0.329 (p<0.001) |
| CCVGAE vs scVI（scRNA）| 总体 UMAP score | +0.219 |
| CCVGAE vs PeakVI（scATAC）| core quality | +0.273 |
| GAT-VAE 的 trade-off | scRNA NMI/ARI↑ 但 manifold dim -0.016，core quality -0.013（均 p≥0.05）| CCVGAE 组合后恢复 |

---

## 10. 数据 manifest 文件信息

- 输出文件：`/home/zeyufu/LAB/scCCVGBen/data/revised_methodology_manifest.csv`
- 总行数：172（57 scRNA 含 2 case-study 专用 + 115 scATAC）
- 注：55 benchmark scRNA + 2 case-study-only（CSD_1014, UCB_timeseries）= 57 scRNA 行；115 scATAC 行

---

## 11. Executor Batch 2 需 awareness 的 Top 3 事项

### (1) 数据集目标数量是 170，非 200
revised CCVGAE 使用 55 scRNA + 115 scATAC = 170。  
scCCVGBen 原计划 100+100=200 **与 revised 不符**。  
Batch 2 在重建 datasets.csv 时应以 170 为锚点，或明确记录为何扩充至 200（需 justification）。

### (2) scATAC 路径在 Desktop/scATAC-25100/，非 Downloads/
115 个 scATAC 数据集的原始 h5 文件在 `/home/zeyufu/Desktop/scATAC-25100/{N}-{GSE}/{sample}.h5`，  
supplement notebook 会将其转换为 h5ad 存入 Downloads/ATAC_data/。  
on-host inventory scan（T2/T3）应优先扫描这两个目录。

### (3) GAT-VAE 有 clustering-vs-geometry trade-off，CCVGAE 组合后消除
单独使用 GAT-VAE 在 scRNA 上 NMI/ARI 稍升但 manifold geometry 略降（p≥0.05）。  
这个 trade-off 在 scCCVGBen baseline 计算中需正确实现（消融四变体 VAE/CenVAE/CouVAE/GAT-VAE/CCVGAE），  
且 revised 版要求用 10-run k-means + lowest inertia（非单次），seed=42 用于 UMAP/t-SNE。
