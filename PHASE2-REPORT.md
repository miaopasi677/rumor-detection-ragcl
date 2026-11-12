# 第二阶段报告：AD-GSCL 与 BiGCN 对比实验

生成日期：2026-11-12 · 项目：图驱动谣言检测（客户委托）· 阶段：第一步复现与对比

> 本报告只记录本机真实运行产生的复现与对比结果。不含推测值，未为提升指标调整任何超参，不承诺发表或录用。

## 1. 目标与范围

在已完成的 RAGCL 主 baseline 复现基础上，补充两个对比模型：

| 模型 | 定位 | 来源 |
|---|---|---|
| AD-GSCL | 与 RAGCL 同课题组的新一代方法 | arXiv 2508.07205，官方代码 `819fe342…` |
| BiGCN | 经典图模型对照，仅放对比表 | AAAI'20，官方代码 `5779f2cf…` |

本阶段**不做模型改进、不做创新点验证、不做查新**；只交付可复现的代码、原始日志、逐次运行结果与对比结论。

## 2. 统一实验协议

| 项 | 值 |
|---|---|
| 数据集 | DRWeiboV3（官方 `Data.zip`，6037 帖；标签 3185 / 2852） |
| 划分 | 802（RAGCL `Main/sort.py`，`random.seed(1234)`；train 4829 / val 1 / test 1208） |
| 重复 | 10 次独立 run × 100 epoch |
| 主指标 | 每 run 最后 10 个 epoch 的 Test ACC 均值（官方 `record_sup.py` 的 `cal_mean=-10`） |
| 离散度 | 10 run 的总体标准差（n，官方口径） |
| 95% CI | `mean ± 1.96 × SD/√10` |

`val` 仅 1 条样本，Val ACC 不具参考意义，主指标只看 Test ACC。官方 `sort.py` 固定 `random.seed(1234)`，因此 10 个 run 是**同一划分、不同随机初始化**，不是 10 次重新划分。

## 3. 关键口径决策（必须随结果说明）

### 3.1 AD-GSCL：与 RAGCL 严格可比

官方默认 `jieba` 分词 + `UWeiboV2` 无标签语料，与 RAGCL 输入不一致。为让差异**只来自模型结构与损失**，改为 `--tokenize_mode naive --unsup_dataset UWeibo`，复用 RAGCL 同一份 `w2v_DRWeiboV3_naive_20000_200.model`。官方源码未改。AD-GSCL 官方默认设置未使用，这一点在任何汇报中都需注明。

### 3.2 BiGCN：统一到 DRWeiboV3 的适配版

官方 BiGCN 基于 Weibo / TD-RvNN 格式、5000 维词袋、5-fold CV、200 epoch + EarlyStopping、torch 1.4 / PyG 1.3.2，与本次协议完全不同。经负责人确认，采用「统一到 DRWeiboV3」口径：

- **未改**：用 `ast` 从官方 `BiGCN_Weibo.py` 原样抽取 `TDrumorGCN` / `BUrumorGCN` / `Net` 三个类；`Net(5000,64,64)`、lr `0.0005`、weight_decay `1e-4`、batch 16、BU conv `lr/5`、`nll_loss` 全部保留。官方源码文件没有任何改动。
- **已适配**：TRAIN-only 文本上用 jieba 自建 top-5000 词袋（**不是作者原始词表**）；802 划分代替 5-fold CV；固定 100 epoch 代替 200 epoch + EarlyStopping；`num_workers 10→0`、`cuda:1→cuda:0`。

因此 BiGCN 数字**不能**与论文中 Weibo 上的 BiGCN 数值直接比较，也不能据此断言「BiGCN 模型弱于 RAGCL」。

## 4. 结果

| 模型 | 完整 run | Test ACC (mean ± SD) | 95% CI | macro-F1 |
|---|---:|---:|---:|---:|
| RAGCL (AAAI'24)，主 baseline | 10/10 | **0.8812 ± 0.0045** | 0.8784–0.8840 | 0.8801 ± 0.0046 |
| AD-GSCL，新一代对比 | 10/10 | **0.8817 ± 0.0050** | 0.8786–0.8848 | 0.8811 ± 0.0051 |
| BiGCN (AAAI'20)，经典对照·适配版 | 10/10 | **0.8719 ± 0.0030** | 0.8701–0.8738 | 0.8717 ± 0.0029 |

BiGCN 逐 run 后 10 epoch Test ACC：

```
0.8751  0.8708  0.8711  0.8727  0.8684  0.8671  0.8737  0.8740  0.8695  0.8770
```

独立重算：均值 `0.871940`、总体 SD `0.002957`，与汇总脚本输出一致。10 份官方 JSON 全部 `epochs_recorded=101`，全部 `exit_code=0`。

**读数说明**

- RAGCL 与 AD-GSCL 差值仅 `0.0005`，远小于两者 run 间 SD（0.0045 / 0.0050），应视为**基本持平**，不宣称「AD-GSCL 更好」。
- BiGCN 低约 `0.0093–0.0098`（约 2–3 倍 SD），但输入特征与评估协议都不同，属适配口径，**不作论文级优劣结论**。
- RAGCL 本机值 `0.8812` 低于论文约 `0.894`：可能来自 20k 固定种子 UWeibo 子集（非全量池）、gensim / scikit-learn 兼容版本、Windows + cu113、官方代码未固定模型初始化种子。全程未为涨点调参。
- CI 统一按总体标准差（n）的 SEM 计算。`RAGCL/logs/final_ragcl_drweibo_full_metrics.json` 内另有一份按样本标准差（n−1）算的稍宽 CI `0.8782–0.8841`，仅为统计约定差异，均值与 SD 完全一致。

## 5. 真实性证据清单

| 内容 | 路径 |
|---|---|
| AD-GSCL 10 份官方指标 JSON | `AD-GSCL/Log/*.json` |
| AD-GSCL 逐 run 命令、耗时、exit code | `adgscl_manifest.json` |
| AD-GSCL 汇总 | `AD-GSCL/logs/final_adgscl_drweibo_summary.json` |
| BiGCN 10 份官方格式 JSON + 10 份控制台日志 | `BiGCN/Log/*.json`、`BiGCN/logs/bigcn_run*.log` |
| BiGCN 驱动时间线（含中断与恢复） | `BiGCN/logs/resume_bigcn_driver.log` |
| BiGCN 逐 run 命令、耗时、exit code | `bigcn_manifest.json` |
| BiGCN 汇总（逐 run / CI / 适配说明） | `BiGCN/logs/final_bigcn_drweibo_summary.json` |
| 三模型对比表与机器可读明细 | `COMPARISON-DRWEIBOV3.md`、`three_model_comparison.json` |
| 全程复现记录（环境、数据审计、故障与修复） | `RAGCL-LOCAL-REPRODUCTION.md` |

复现脚本（均为我方新增，不改官方模型数学）：`run_adgscl_safe.py`、`resume_adgscl_runs.py`、`build_bigcn_drweibo.py`、`run_bigcn_drweibo.py`、`resume_bigcn_runs.py`、`compare_three_models.py`。

## 6. 工程记录（不影响结果口径）

1. **安全包装器**：官方 `sort.py` 的 `dataset_makedirs()` 内 `shutil.rmtree()` 会批量删除 split 文件，触发本机安全删除保护。包装器只替换该函数，复用固定 802 划分并仅删除 3 个 `processed/data.pt` 缓存。
2. **稀疏特征缓存**：BiGCN 6037 张图的 5000 维 dense 特征全量缓存需约 7.6 GB，改为只缓存 scipy CSR（约 40 MB），按 batch 转 dense。
3. **数值等价的运行时适配**：官方 `forward()` 的 `th.zeros(n,5000).to(device)` 先在 CPU 分配再搬 GPU；并发时触发 OOM。改为加载官方类时传入薄代理让 `zeros()` 直接在 GPU 分配。`torch.zeros(n,m,device=cuda)` 与 `torch.zeros(n,m).to(cuda)` 数值完全相同，官方源码文件未改。
4. **并发实测后回退单进程**：3–4 进程 OOM；2 进程虽把 GPU 利用率从约 12% 提到 85%，但 commit charge 达 30.3 GB（物理内存 15.7 GB），RSS 被换出、单 epoch 从 62 s 恶化到 >141 s，总吞吐低于串行。最终固定 1 worker。
5. **中断与断点续跑**：2026-11-11 的 3-worker 批次、2026-11-12 00:28 的 2-worker 批次均无 run 落盘，不计入统计；00:55 的单进程批次完成 run0–6；10:54 因负责人做 GPU 测试占用内存，run7 报 `_ArrayMemoryError: 2.94 MiB` 崩溃、run8 仅到 epoch 000 后被停止，二者无完整 JSON、不计入统计；11:55 从断点续跑 run7–9，16:02 全部完成。所有中途结果均未复用。
6. **统计约定**：SD 用总体标准差（n，与官方 `record_sup.py` 一致）；CI 用同口径 SEM。

## 7. 结论与边界

- 第一阶段两步实验部分完成：RAGCL 主 baseline 跑通并完成 10 次复现，AD-GSCL / BiGCN 各完成 10 次同协议对比。
- 结果为**本机复现实验结果**，不是论文数值的重现；不宣称「复现到论文结果」。
- BiGCN 为适配版对照，必须与口径说明一起呈现。
- 本阶段不构成任何发表、录用或 CCF-B 级别保证。

## 8. 下一步

- 创新方向筛选与查新（候选：时间感知评测、LLM 生成评论注入鲁棒性等，**尚未确认是否已有工作**）。
- 与客户确认后续方案、署名与代码归属。
