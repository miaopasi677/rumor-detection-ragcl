# DRWeiboV3 三模型对比（真实运行日志）

生成时间：2026-11-12T16:05:10

- 数据集：DRWeiboV3（官方 Data.zip）
- 划分：802（RAGCL Main/sort.py, random.seed(1234)；train 4829 / val 1 / test 1208）
- 协议：10 runs x 100 epochs；主指标 = 每 run 最后 10 个 epoch 的 Test ACC 均值（官方 record_sup.py cal_mean=-10）；SD 为 10 run 的总体标准差

## 结果表

| 模型 | 角色 | 输入特征 | 完整 run | Test ACC (mean ± SD) | 95% CI | macro-F1 |
|---|---|---|---|---|---|---|
| RAGCL (AAAI'24) | 主 baseline | 官方 word2vec（naive 分词，UWeibo 20k 固定子集，200 维） | 10/10 | 0.8812 ± 0.0045 | 0.8784-0.884 | 0.8801 ± 0.0046 |
| AD-GSCL | 新一代对比 | 与 RAGCL 完全相同的 word2vec / 分词 / 无标签语料 | 10/10 | 0.8817 ± 0.0050 | 0.8786-0.8848 | 0.8811 ± 0.0051 |
| BiGCN (AAAI'20) | 经典对照（适配版） | 自建 jieba top-5000 词袋（TRAIN-only），非作者原词表 | 10/10 | 0.8719 ± 0.0030 | 0.8701-0.8738 | 0.8717 ± 0.0029 |

## 口径与限制（必须随结果一起说明）

1. **RAGCL 与 AD-GSCL 严格可比**：同一份 `w2v_DRWeiboV3_naive_20000_200.model`、同一 naive 分词、同一 20k UWeibo 子集、同一 802 划分、同一 10×100 协议；差异只来自模型结构与损失。AD-GSCL 官方默认 `jieba + UWeiboV2` 未使用。
2. **BiGCN 是适配版，不是官方复现**：官方基于 Weibo 的 5000 维词袋、5-fold CV、200 epoch + EarlyStopping；本实验为统一到 DRWeiboV3，改用 TRAIN-only jieba top-5000 词袋、802 划分、固定 100 epoch、无早停。官方 `Net` 三个类、lr=0.0005 / weight_decay=1e-4 / batch=16 / hidden=64 / BU conv lr÷5 均未改。因此 BiGCN 数字不能与论文中 Weibo 上的数值直接比较。
3. **RAGCL 本机值低于论文**：0.8812 vs 论文约 0.894。可能来源：20k 固定种子UWeibo 子集而非全量池、gensim/sklearn 兼容版本、Windows + cu113、官方代码未固定模型初始化种子。全程未为涨点调参。
4. `val` 只有 1 条样本，Val ACC 不具参考意义；主指标只看 Test ACC。
5. 所有数字来自 `RAGCL/Log/*.json`、`AD-GSCL/Log/*.json`、`BiGCN/Log/*.json` 真实日志，机器可读明细见 `three_model_comparison.json`。
6. **统计约定**：表中 SD 为 10 run 的总体标准差（n，官方 `record_sup.py` 口径）；95% CI 用同一 SD 的 SEM 计算，即 `mean ± 1.96 × SD/√n`。`RAGCL/logs/final_ragcl_drweibo_full_metrics.json` 中另存有一份按样本标准差（n−1）计算的 CI（0.8782–0.8841），比本表略宽；两者仅为统计约定差异，不改变均值、SD 或任何结论。

## 各模型备注

- **RAGCL (AAAI'24)**：官方 main(sup).py 源码未修改；复现值低于论文 0.894，如实记录。
  - 来源日志 5 份。
- **AD-GSCL**：为严格可比，未用官方默认 jieba + UWeiboV2，改为 naive + UWeibo；仅模型/损失不同。
  - 来源日志 10 份。
- **BiGCN (AAAI'20)**：官方 Net/TDrumorGCN/BUrumorGCN 类未改；5-fold CV、EarlyStopping、200 epoch 被统一协议替换。与论文数字不可直接比较。
  - 来源日志 10 份。
