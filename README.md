# RAGCL 复现工程（图驱动谣言检测 · 第一步）

本仓库保存**本机从零复现 RAGCL** 所需的脚本、精确依赖、真实运行日志和结果汇总。官方训练源码一行未改，所有数字均由实际命令产生；不含任何推测值或调参涨点。

- 论文：*Propagation Tree Is Not Deep: Root Alignment Graph Contrastive Learning for Rumor Detection*（AAAI 2024）
- 官方代码：<https://github.com/CcQunResearch/RAGCL>
- 固定版本：`RAGCL@bda6a1885cd00fe174ac8fae4a629f1289fa496b`（未改动）

## 真实结果（DRWeiboV3，10 runs × 100 epochs）

| 指标 | 均值 | 总体 SD |
|---|---:|---:|
| **Test ACC（主指标）** | **0.8812** | **0.0045** |
| macro-F1 | 0.8801 | 0.0046 |
| macro-Precision | 0.8857 | 0.0047 |
| macro-Recall | 0.8789 | 0.0045 |

- 主指标口径：官方 `Log/record_sup.py` 的 `cal_mean=-10`，即每 run 最后 10 个 epoch 的 test acc 均值，再对 10 run 取均值；SD 为总体标准差。
- ACC 单 run 范围 0.8763–0.8912，95% CI `[0.8782, 0.8841]`。
- 最佳单 epoch ACC 均值 0.8956（最高 0.9015）**仅供参考，不能当作论文结果汇报**。
- 交接文档中的论文值 0.894、另一台 Linux VM 的 0.869、BiGCN 0.866 **均不是本机产生**，且论文原表尚未核对，不可当作对照结论。
- 与论文的差距可能来自：UWeibo 20k 固定种子子集（非官方全量池）、gensim / scikit-learn 兼容版本、Windows + CUDA 11.3、官方代码模型初始化未设随机种子。全程未为涨点调整任何超参。

详见 `RAGCL-LOCAL-REPRODUCTION.md`（完整复现记录）与 `RAGCL/logs/final_ragcl_drweibo_full_metrics.json`（机器可读逐 run 明细、分组统计、CI）。

## 目录结构

布局与本机工作目录一致（脚本用 `Path(__file__).parent / "RAGCL"` 定位官方代码，因此脚本必须放在仓库根目录）。

```
*.py                     我方编写的复现 / 运行 / 汇总脚本（不改官方模型逻辑）
RAGCL/Log/               官方 main(sup).py 写出的 *.json + *.log（原始指标证据）
RAGCL/logs/              最终 10-run 指标、逐 run 汇总、控制台日志、烟雾测试与 4-run 部分结果
parallel_shared/logs/    共享缓存并行 worker 日志与 status.json
requirements-gpu.txt     精确依赖版本
resume_manifest.json     10 个 run 的来源日志、命令与耗时
RAGCL-LOCAL-REPRODUCTION.md  完整复现与环境记录
```

`RAGCL/Log/` 中部分短 `.log` 是被中断或兼容性失败留下的残日志，**没有对应 JSON，不计入任何统计**；仅下列 5 份 JSON 组成最终 10 run：

- `2026-11-09 22-02-34.json`（4 run，单进程）
- `2026-11-10 18-48-17.json`、`2026-11-10 19-16-23.json`（各 1 run，断点补跑）
- `2026-11-10 19-58-57.json`、`2026-11-10 19-59-04.json`（各 2 run，共享缓存并行）

口径混合这一事实已如实记录：4 run 单进程、2 run 独立进程、4 run 共享确定性图缓存并行；三组均值差异均在 SD 量级内。若需严格单进程单一 JSON，可按复现文档 5.1 重跑（约 5–6 小时）。

## 本仓库不包含的内容

出于体积与授权考虑，以下内容**不在仓库内**，需按复现文档自行获取：

- 官方 RAGCL / AD-GSCL / BiGCN 源码（请从各自官方仓库 clone，版本见复现文档）
- `Data.zip`（DRWeiboV3、Weibo、Twitter15/16，约 87 MB）与解压后的数据集
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
- [ ] 第二步：AD-GSCL 与 BiGCN 对比、候选创新方向查新

数据侧已确认：四个数据集中**只有 DRWeiboV3 带完整时间戳**（源帖发出 0.5 小时内有首条评论的占 74.95%，1 小时 79.53%，6 小时 88.79%，24 小时 95.44%；6037 帖全部有评论且时间可解析）。超早期检测方向只能在 DRWeiboV3 上开展，不能默认四个数据集都可时间切分。
