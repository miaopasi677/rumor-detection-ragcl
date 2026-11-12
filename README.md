# 图驱动谣言检测 · 第一阶段复现与对比工程

本仓库保存本机从零复现 RAGCL、并完成 AD-GSCL / BiGCN 对比所需的脚本、精确依赖、**真实运行日志**与结果汇总。官方训练源码一行未改，所有数字均由实际命令产生；不含推测值、未为涨点调参、不承诺发表或录用。

- 主 baseline 论文：*Propagation Tree Is Not Deep: Root Alignment Graph Contrastive Learning for Rumor Detection*（AAAI 2024）
- 官方代码：<https://github.com/CcQunResearch/RAGCL>，固定 `bda6a1885cd00fe174ac8fae4a629f1289fa496b`

## 最终结果（DRWeiboV3 · 802 划分 · 各 10 runs × 100 epochs）

| 模型 | 完整 run | Test ACC (mean ± SD) | 95% CI | macro-F1 |
|---|---:|---:|---:|---:|
| RAGCL（主 baseline） | 10/10 | **0.8812 ± 0.0045** | 0.8784–0.8840 | 0.8801 ± 0.0046 |
| AD-GSCL（新一代对比） | 10/10 | **0.8817 ± 0.0050** | 0.8786–0.8848 | 0.8811 ± 0.0051 |
| BiGCN（经典对照·适配版） | 10/10 | **0.8719 ± 0.0030** | 0.8701–0.8738 | 0.8717 ± 0.0029 |

主指标口径：官方 `Log/record_sup.py` 的 `cal_mean=-10`，即每 run 最后 10 个 epoch 的 Test ACC 均值，再对 10 run 取均值；SD 为总体标准差（n），CI 为 `mean ± 1.96 × SD/√10`。`val` 只有 1 条样本，Val ACC 无参考意义。

- RAGCL 与 AD-GSCL 差值 0.0005，远小于各自 SD，视为**持平**。
- BiGCN 是**适配版**：官方 `Net` 类与超参未改，但改用自建 train-only jieba top-5000 词袋、802 划分、固定 100 epoch，代替作者词表 / 5-fold CV / EarlyStopping。**不能与论文 Weibo 数值直接比较**。
- 本机 RAGCL 0.8812 低于论文约 0.894，如实记录；差距可能来自 20k 固定种子 UWeibo 子集、兼容库版本、Windows + cu113、官方未固定初始化种子。未做任何针对性调参。
- 结果为复现实验结果，不宣称「复现到论文结果」，不构成发表/录用保证。

详见：

| 文档 | 内容 |
|---|---|
| `PHASE2-REPORT.md` | **第二阶段报告**：协议、口径决策、结果、真实性证据清单、工程记录 |
| `COMPARISON-DRWEIBOV3.md` | 三模型对比表与必须随结果说明的口径限制 |
| `three_model_comparison.json` | 机器可读对比明细 |
| `RAGCL-LOCAL-REPRODUCTION.md` | 全程复现与环境记录（数据审计、故障与修复、最终指标） |
| `RAGCL/logs/final_ragcl_drweibo_full_metrics.json` | 第一步逐 run 指标、分组统计、CI |

## 目录结构

布局与本机工作目录一致（脚本用 `Path(__file__).parent / "RAGCL"` 定位官方代码，因此脚本必须放在仓库根目录）。

```
*.py                     我方编写的复现 / 运行 / 汇总脚本（不改官方模型逻辑）
RAGCL/Log/               官方 main(sup).py 写出的 *.json + *.log（原始指标证据）
RAGCL/logs/              第一步 10-run 指标、逐 run 汇总、控制台日志
AD-GSCL/Log/             AD-GSCL 10 份官方指标 JSON
AD-GSCL/logs/            AD-GSCL 控制台日志与汇总
BiGCN/Log/               BiGCN 10 份官方格式 JSON（+ 2 份烟雾测试）
BiGCN/logs/              BiGCN 控制台日志、驱动时间线、汇总
parallel_shared/logs/    共享缓存并行 worker 日志
*_manifest.json          逐 run 命令、耗时、exit code、来源日志
requirements-gpu.txt     精确依赖版本
```

`RAGCL/Log/` 中部分短 `.log` 是被中断或兼容性失败留下的残日志，**没有对应 JSON，不计入任何统计**；第一步最终 10 run 来自下列 5 份 JSON：

- `2026-11-09 22-02-34.json`（4 run，单进程）
- `2026-11-10 18-48-17.json`、`2026-11-10 19-16-23.json`（各 1 run，断点补跑）
- `2026-11-10 19-58-57.json`、`2026-11-10 19-59-04.json`（各 2 run，共享缓存并行）

口径混合已如实记录：4 run 单进程、2 run 独立进程、4 run 共享确定性图缓存并行，三组均值差异均在 SD 量级内。若需严格单进程单一 JSON，可按复现文档 5.1 重跑（约 5–6 小时）。

第二步的关键口径决策：AD-GSCL 官方默认 `jieba + UWeiboV2` 未使用，改为与 RAGCL 完全相同的 `naive + UWeibo` 及同一份 word2vec，使差异只来自模型/损失；BiGCN 按负责人选择统一到 DRWeiboV3。细节见 `PHASE2-REPORT.md` 第 3 节。

## 本仓库不包含的内容

出于体积与授权考虑，以下内容**不在仓库内**，需按复现文档自行获取：

- 官方 RAGCL / AD-GSCL / BiGCN 源码（从各自官方仓库 clone，版本见下表）
- `Data.zip`（DRWeiboV3、Weibo、Twitter15/16，约 87 MB）与解压后的数据集、BiGCN 生成的 6037 个 `.npz` 图缓存
- UWeibo 无标签语料与 `word2vec` 模型、PyTorch CUDA wheel（约 2 GB）
- Python 虚拟环境、论文 PDF 与论文包

官方代码版权归原作者所有；本仓库只提交我方新增的运行 / 汇总脚本与真实日志。

## 固定版本（供复现）

| 仓库 | commit |
|---|---|
| RAGCL | `bda6a1885cd00fe174ac8fae4a629f1289fa496b` |
| AD-GSCL | `819fe342aad021a9418daba63a6b7788b972009b` |
| BiGCN | `5779f2cf86a4649651f2466b43c73caa6ed7463e` |
| UWeibo | `c6db716544ee487324961eb2548ef9d027999b74` |
| DRWeibo | `9f90a7ec12900c9d6fd9a503c2232092363fa758` |

`Data.zip` SHA-256：`53864a8c8cd3946a46b30d6426774e1906415125ebc4613505d899175f735c65`

## 环境

Windows 11 · NVIDIA RTX 3080 Laptop 16 GB（驱动 471.41 / CUDA 11.4）· Python 3.10 ·
PyTorch `1.12.0+cu113` · PyG 2.1.0 · gensim 4.3.2 · scikit-learn 1.0.2（`labels=True` 旧 API 兼容需要）。
精确版本见 `requirements-gpu.txt`。

## 状态

- [x] 第一步：RAGCL 复现，10 run 完成，结果已复核
- [x] 第二步：AD-GSCL 与 BiGCN 对比，各 10 run 完成，报告已提交
- [ ] 创新方向筛选与查新（未开展；候选方向均未经查新确认）

数据侧已确认：四个数据集中**只有 DRWeiboV3 带完整时间戳**（源帖发出 0.5 小时内有首条评论的占 74.95%，1 小时 79.53%，6 小时 88.79%，24 小时 95.44%；6037 帖全部有评论且时间可解析）。超早期检测方向只能在 DRWeiboV3 上开展，不能默认四个数据集都可时间切分。
