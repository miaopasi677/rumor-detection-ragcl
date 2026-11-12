# RAGCL 本地 GPU 复现记录（Windows / RTX 3080 Laptop）

> 开始时间：2026-11-09 晚；**2026-11-10 21:27 完成 10 run × 100 epoch 正式实验**。本文记录本机从零复现的每一步。所有结果只填写真实命令产生的日志；未完成的实验明确标为“进行中/待执行”，不使用推测值或编造值。最终结果见 5.6，机器可读明细见 `RAGCL/logs/final_ragcl_drweibo_full_metrics.json`。

## 1. 交接状态校正

此前 Linux VM 已完成 1 run × 30 epoch 的快速验证，但本机工作区最初为空，不能直接复用其 `.venv` 或日志。本机重新下载并固定了官方代码、数据及对比模型。

用户后续补充的材料已经齐备：

- `Data.zip`：百度网盘官方数据包，84MB，完整性测试通过。
  - SHA-256：`53864a8c8cd3946a46b30d6426774e1906415125ebc4613505d899175f735c65`
  - 顶层目录：`DRWeiboV3`、`Weibo`、`Twitter15-tfidf`、`Twitter16-tfidf`
- 微信聊天中的 DDGCN PDF、两篇综述、01/02/03 三个论文包均已复制到 `papers/`。
- 官方 AD-GSCL 仓库链接已核实为 `https://github.com/CcQunResearch/AD-GSCL`。

因此，原交接总结中“Weibo 数据拿不到”“客户论文包还没发”两项卡点均已解除。

## 2. 本机环境

| 项目 | 实际值 |
|---|---|
| 系统 | Windows 11，Git Bash |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU，16GB，compute capability 8.6 |
| 驱动 / 驱动支持 CUDA | 471.41 / CUDA 11.4 |
| Python | uv 管理的 CPython 3.10.22 |
| CPU 环境 | `RAGCL/.venv`：`torch==1.12.0+cpu`，仅用于 word2vec |
| GPU 环境 | `RAGCL/.venv-gpu`：`torch==1.12.0+cu113`，用于正式训练 |
| 官方要求 | PyTorch 1.12.0、PyG 2.1.0、gensim 4.0.1 |

已安装的替代/兼容版本：`gensim==4.3.2`（4.0.1 无 Python 3.10 Windows wheel）、`scipy==1.11.4`（保留 gensim 所需 `triu`）、`numpy==1.23.5`、`torch-geometric==2.1.0`（PyPI 标记 yanked：Missing ninja templates，但官方要求此版本）、`torch-scatter==2.1.0+pt112cu113`、`torch-sparse==0.6.16+pt112cu113`、`torch-cluster==1.6.0+pt112cu113`、`torch-spline-conv==1.2.1+pt112cu113`。

首次烟雾测试发现 `scikit-learn==1.3.2` 拒绝官方代码使用的旧 API `labels=True`。为保持官方训练代码一行不改，GPU 环境降为 `scikit-learn==1.0.2`，并已用最小样例验证旧 API 正常。

> 官方 PyTorch CUDA wheel 2,143,782,114 字节，PyTorch 官方源单连接极慢并曾卡住一次；改用阿里云镜像的 16 段并发 Range 下载（`download_torch_wheel_parallel.py`）。最终 SHA-256 为 `0ac18d1d8511533270f38e62243a0ab6aed9863d4b89ddedc1ef5fec87a0fb48`，与官方索引完全一致。GPU 环境实测 `torch.cuda.is_available()==True`，CUDA scatter_add 返回 `[2.0, 2.0]`。

## 3. 代码与数据版本

| 来源 | 固定版本 |
|---|---|
| RAGCL | `bda6a1885cd00fe174ac8fae4a629f1289fa496b` |
| DRWeibo（GitHub 备用数据） | `9f90a7ec12900c9d6fd9a503c2232092363fa758` |
| UWeibo | `c6db716544ee487324961eb2548ef9d027999b74` |
| AD-GSCL | `819fe342aad021a9418daba63a6b7788b972009b` |
| BiGCN（原作者仓库） | `5779f2cf86a4649651f2466b43c73caa6ed7463e` |

正式实验使用 `Data.zip` 中的 `DRWeiboV3`，不采用 GitHub DRWeibo 的字符串字段转换版。官方包中 `parent`、`comment id` 已是整数，并已包含四种中心性，因此无需修改数据格式，也无需运行 `centrality.py`。

### 3.1 数据完整性审计

| 数据集 | 源帖数 | 评论数 | 标签 | 时间字段 |
|---|---:|---:|---|---|
| DRWeiboV3 | 6037 | 373,437 | 0:3185 / 1:2852 | 源帖 6037 / 评论 373,437，全部可解析 |
| Weibo | 4664 | 840,611 | 0:2351 / 1:2313 | 全部缺失 |
| Twitter15-tfidf | 1490 | 74,918 | 四类，各类 370–374 | 全部缺失 |
| Twitter16-tfidf | 818 | 40,176 | 四类，各类 201–207 | 全部缺失 |

- 发布包 Weibo 平均每图约 181 个节点（840,611 / 4664 + 1），DRWeiboV3 约 63 个，比例约 2.9。
- 论文统计表写 Weibo 平均 803.5 条回复，与发布包实际数量存在明显差异；后续不能把论文统计值和发布包实测值混为一谈。
- 只有 DRWeiboV3 能做严格的时间阈值评测；Weibo/Twitter15/16 的 tf-idf 发布包无时间戳。

### 3.2 DRWeiboV3 超早期检测可行性

按 `%y-%m-%d %H:%M` 解析全部源帖与评论时间：无解析失败、无负时间差。首条评论落入各时间窗的源帖数：

| 检查点 | 源帖数 | 占 6037 条比例 |
|---|---:|---:|
| 30 分钟 | 4525 | 0.7495 |
| 1 小时 | 4801 | 0.7953 |
| 6 小时 | 5360 | 0.8879 |
| 24 小时 | 5762 | 0.9544 |

首条评论延迟中位数 0.05 小时，75 分位 0.517 小时，90 分位 8.283 小时，最大 15172.283 小时。该数据足以支撑 30 分钟/1 小时/6 小时/24 小时检查点，但需注意约 25% 源帖在 30 分钟内没有任何评论。

## 4. 与官方训练设置的偏差

- UWeibo 仓库含 209,549 条 JSON，磁盘仅约 10–20GB 可用，故用 `prepare_uweibo_subset.py` 以固定种子 1234 抽取 20,000 条（约 556MB），并生成清单 `RAGCL/Data/UWeibo/uweibo_subset_manifest.json`。
- 这与官方“从全量 UWeibo 随机抽取 20,000 条”的抽样池不完全相同，但数量、语料来源、训练代码和 30 epochs 保持一致；清单含源 7z SHA-256，可复现本次抽样。
- 命令显式使用 `--unsup_dataset UWeibo`，因为官方 UWeibo 仓库目录名是 `UWeibo`，而 RAGCL 默认值是 `UWeiboV1`。
- word2vec 使用官方 `Main/word2vec.py`，`epochs=30`、`workers=12`、`vector_size=200`、`min_count=5`、`sg=1`；训练脚本为 `train_ragcl_word2vec.py`。
- nltk 在导入 `utils.py` 时尝试下载 `punkt` 曾失败，但 `tokenize_mode=naive` 的中文分支不调用 punkt，不影响本次训练。
- 官方 `sort.py` 固定 `random.seed(1234)`，10 个 run 实际是同一数据划分、不同随机初始化，并非 10 次独立随机划分；保持官方代码不改。
- 802 划分下验证集只有 1 条样本，Val ACC 无参考意义；以 Test ACC 和官方 JSON 中的后 10 epoch 平均为准。
- 本机安全保护禁止一次删除 6047 个旧 split 文件。`run_ragcl_safe.py` 仅替换 `Main.sort.dataset_makedirs`：保留 raw 目录，让固定种子生成的同名 split JSON 被官方 `write_post()` 覆盖；只删除 train/val/test 三个旧的 `processed/data.pt`，强制每 run 重新处理图数据。RAGCL 源码、数据划分、训练与评估逻辑均不变，每轮仅 3 次单文件删除，不触发批量删除保护，也不会累积 490MB 级的 split 归档。

## 5. 执行状态

- [x] 官方 RAGCL、DRWeibo、UWeibo、AD-GSCL、BiGCN 代码下载与版本固定
- [x] `Data.zip` 完整性校验、四数据集审计、DRWeiboV3 时间戳审计
- [x] Python 3.10 CPU/GPU 双环境与官方依赖安装，依赖固定在 `requirements-gpu.txt`
- [x] CUDA 版 PyTorch 镜像下载、SHA-256 校验及 GPU/CUDA scatter 验证
- [x] word2vec（30 epochs + 20k UWeibo）：1,410,307 句、词表 6964、990.9 秒
- [x] 2-epoch GPU 烟雾测试：111 秒、退出码 0；epoch 1/2 的 Test ACC 为 0.8303/0.8295
- [x] DRWeiboV3 正式 10 runs × 100 epochs：10/10 全部完整落盘、退出码均为 0（单进程 4 + 补跑 2 + 共享缓存并行 4）
- [x] 4 个完整 run 的部分结果已归档（`RAGCL/logs/partial_4runs_summary.json`），非最终统计
- [x] 汇总完整 10-run 结果并与论文值/此前 VM 结果对照（见 5.6）
- [ ] 第二步：AD-GSCL + BiGCN 对比及查新（代码已下载并固定 commit，尚未运行）

### 5.1 关键命令

```bash
# word2vec（CPU 环境）
RAGCL/.venv/Scripts/python.exe train_ragcl_word2vec.py

# 烟雾测试（GPU 环境）
RAGCL/.venv-gpu/Scripts/python.exe -u run_ragcl_safe.py \
  --dataset DRWeiboV3 --unsup_dataset UWeibo --runs 1 --epochs 2 --gpu 0

# 正式实验（GPU 环境，单进程严格顺序 10 runs）
RAGCL/.venv-gpu/Scripts/python.exe -u run_ragcl_safe.py \
  --dataset DRWeiboV3 --unsup_dataset UWeibo --runs 10 --epochs 100 --gpu 0
```

烟雾测试日志：`RAGCL/Log/2026-11-09 21-29-05.log` 与同名 JSON；汇总：`RAGCL/logs/smoke_summary.json`。正式 10-run 汇总由 `resume_ragcl_runs.summarize()` 生成 `RAGCL/logs/final_ragcl_drweibo_summary.json`，完整指标由 `compute_final_metrics.py` 从原始 JSON 重算生成 `RAGCL/logs/final_ragcl_drweibo_full_metrics.json`（可重复执行复核）。任何结果与本文件摘要不一致时，以原始 `Log/*.json` 为准。

### 5.2 并行尝试（不计入最终结果）

单进程 GPU 利用率约 18%，曾尝试 `run_parallel_full.py` 用 3 个隔离副本跑 4/3/3 个 run。两个 worker 在同时构建图缓存时发生 `torch.save` 内存分配失败并退出 `3221226505`，第三个独立运行正常。由于本机可用内存不足，放弃并行结果，所有失败 JSON 均不纳入正式统计；保留 `parallel_runs/logs/` 作为环境排查记录。正式结果以官方代码产生的完整 run 日志为准，并逐 run 记录来源（见 5.3）。

### 5.3 单进程中断与分批恢复

- 单进程正式实验 `Log/2026-11-09 22-02-34.json` 于 **2026-11-10 00:05:24** 停止。退出码 `1073807364`（`0x40010004`，进程被外部终止）不是训练代码错误；控制台日志停在数据处理阶段，没有正常的耗时收尾行。
- 官方 `main(sup).py` 第 218–219 行只在 100 个 epoch 全部结束后才把 run 追加进 JSON，因此被中断的第 5 个 run（日志只到 epoch 008）没有落盘记录，**不计入统计**。
- 已落盘 4 个完整 run，每个 `test accs` 长度 101（epoch 0 + 100 个训练 epoch）。
- 恢复方式：`resume_ragcl_runs.py` 以独立 `--runs 1` 进程补齐剩余 6 run，每批最多 2 run；每个 run 保留自己的官方 `Log/*.json`，`resume_manifest.json` 记录命令、耗时与日志路径，可断点续跑。官方代码、参数、802 划分和 word2vec 模型均未改动。
- 口径说明：官方代码只有 `sort.py` 固定 `random.seed(1234)`（划分完全一致），模型初始化未设种子；单进程连续 10 run 与分进程独立 10 run 都是独立随机初始化，但 RNG 来源不同。因此最终报告同时给出「单进程 4 run」与「合并 10 run」两套数字，并逐 run 标注来源日志，不把两者冒充为同一次连续运行。
- 若需要严格单进程、单一 JSON 的 10 run，可重新执行 5.1 的完整命令（约 5–6 小时）；分批方案是为规避长时间后台任务被环境清理。

### 5.4 已落盘的真实部分结果（4 / 10 run）

| Run | 后 10 epoch 平均 Test ACC | 最佳单 epoch |
|---:|---:|---:|
| 0 | 0.8829 | 0.8965 |
| 1 | 0.8790 | 0.8949 |
| 2 | 0.8772 | 0.8949 |
| 3 | 0.8800 | 0.8932 |
| **4-run 均值** | **0.8798 ± 0.0021** | **0.8949（均值）/ 0.8965（最高）** |

- 汇总文件：`RAGCL/logs/partial_4runs_summary.json`；原始指标：`RAGCL/Log/2026-11-09 22-02-34.json` 与同名 `.log`。
- 交接记录所载参考值：RAGCL 论文 DRWeibo 0.894、此前 Linux VM 单 run 0.869、BiGCN 0.866。这些数值来自交接文档，**并非本机本次实验产生**；正式对照要等 10 run 全部完成后再做，不能拿 4-run 均值直接下结论。

### 5.5 剩余 4 run 的加速：共享图缓存并行

- **设备核对**：`nvidia-smi` 只有一块 NVIDIA GPU（index 0 = RTX 3080 Laptop，16GB，WDDM，Bus `01:00.0`）；PyTorch `torch.cuda.device_count()==1`、`cuda:0` 名称同为 RTX 3080、capability 8.6。Intel UHD 是集显、不在 CUDA 设备列表里，`--gpu 0` 不可能指向它；任务管理器的 GPU 0/1 编号与 CUDA 编号不是一回事。
- 实测单 run 1737.4 秒（约 29 分钟），单进程时 GPU 利用率约 17%、显存 1122MB。瓶颈是官方实现每个 epoch 的 4 次全量遍历（训练 + train/val/test 评估）和小图逐张 `data.to(device)`，**不是没用 GPU，也不是沙箱降速**；沙箱只是命令隔离层，实际调用的是 `.venv-gpu` 的 CUDA PyTorch。
- `run_ragcl_shared_cache.py` 只替换 `Main.sort.dataset_makedirs`：保留 3 份 `processed/data.pt`，其余与 `run_ragcl_safe.py` 相同；缓存缺失时直接报错，避免并行重建图触发此前的内存溢出。
- 复用依据：`sort.py` 固定 `random.seed(1234)`，词向量、中心性与图构建均为确定性，因此同一划分的缓存与每 run 重新 `process()` 的结果一致。跳过的是重复确定性计算，数据内容、超参、模型和评估逻辑均未改动。
- `run_shared_cache_parallel.py` 用 2 个 worker（各 2 run）间隔 3 秒启动（仅避免官方日志按秒命名重名），共用同一份缓存；每个 worker 通过 `RAGCL_RAW_ROOT` 把 split JSON 写到独立目录，两个 `sort_dataset()` 不会写同一文件。
- 启动过程曾连续三次被会话的 2 分钟前台超时 SIGTERM 杀掉整个进程组，只留下 2–3 个 epoch 的残日志；改用后台任务启动后稳定运行。残日志没有对应 JSON，不计入统计。
- **并行实测（20:00）**：显存 2063MiB / 16384MiB、GPU 利用率 82%、功耗 46W、62°C，`nvidia-smi` compute apps 出现两个 worker 的 PID（16832 / 26048）；对比单进程 1122MiB / 17%，并行确实把 3080 压起来了。
- 每个 worker 保留自己的官方 `Log/*.json`，来源写入 `resume_manifest.json`；最终 10 run 汇总由 `resume_ragcl_runs.summarize()` 生成，逐 run 标注 `source_log`。

### 5.6 最终 10-run 真实结果（DRWeiboV3，2026-11-10 21:27 完成）

**数据来源**：5 份官方 `RAGCL/Log/*.json`，共 10 个完整 run；逐条复核每个 run 的 `test accs` 长度均为 101（epoch 0 + 100 个训练 epoch），且 `final10_mean_acc` 与从原始 JSON 重算值一致。所有训练进程退出码均为 0；并行两个 worker 分别耗时 5402.7s / 5399.7s。被中断的第 5 个 run 残日志、三次超时残日志、内存溢出并行日志均**未**纳入统计。

**指标口径**：主指标沿用官方 `Log/record_sup.py` 的 `cal_mean=-10`，即最后 10 个 epoch 的 Test ACC 平均；跨 run 标准差为官方同款的总体标准差（population SD）。macro-P/R/F1 是这里对两类逐 epoch 值取算术平均后再对最后 10 epoch 平均，**不是**官方 `test macro metric` 字段。`best_epoch_acc` 仅为次要参考，**不得当作论文结果汇报**（属于挑最好 epoch）。

| 来源日志 | run | 后 10 ep. ACC | macro-F1 | 最佳单 epoch |
|---|---:|---:|---:|---:|
| `2026-11-09 22-02-34.json` | 0 | 0.8829 | 0.8818 | 0.8965 |
| 同上 | 1 | 0.8790 | 0.8780 | 0.8949 |
| 同上 | 2 | 0.8772 | 0.8762 | 0.8949 |
| 同上 | 3 | 0.8800 | 0.8791 | 0.8932 |
| `2026-11-10 18-48-17.json` | 0 | 0.8817 | 0.8808 | 0.8957 |
| `2026-11-10 19-16-23.json` | 0 | 0.8868 | 0.8857 | 0.9015 |
| `2026-11-10 19-58-57.json` | 0 | 0.8797 | 0.8789 | 0.8899 |
| 同上 | 1 | 0.8763 | 0.8749 | 0.8974 |
| `2026-11-10 19-59-04.json` | 0 | 0.8768 | 0.8754 | 0.8907 |
| 同上 | 1 | 0.8912 | 0.8902 | 0.9015 |

| 汇总指标（10 run） | 均值 | 总体 SD |
|---|---:|---:|
| **Test ACC（主指标）** | **0.8812** | **0.0045** |
| macro-F1 | 0.8801 | 0.0046 |
| macro-Precision | 0.8857 | 0.0047 |
| macro-Recall | 0.8789 | 0.0045 |
| F1 / class 0 | 0.8910 | 0.0043 |
| F1 / class 1 | 0.8692 | 0.0051 |
| 最佳单 epoch ACC（次要） | 0.8956 | 0.0041（最高 0.9015） |

ACC 样本标准差 0.00475，SEM 0.00150，正态近似 95% CI **[0.8782, 0.8841]**；单 run 范围 0.8763–0.8912。

**分来源一致性**：单进程 4 run 0.8798、补跑 2 run 0.8843、共享缓存并行 4 run 0.8810。三组均值差异都在总体 SD 量级内，说明复用确定性图缓存没有引入系统性偏差；但口径混合这一事实仍须在对外报告中如实标注（见 5.3）。若客户要求单一进程、单一 JSON 的 10 run，可执行 5.1 的完整命令重跑（约 5–6 小时）。

**与外部参考值的对照**（以下三个数字均来自交接文档，**不是本机本次实验产生**，且论文数值尚未核对原文实验表）：

| 数值 | 来源 | 与本机 0.8812 之差 |
|---|---|---:|
| 0.894 | 交接文档所载 RAGCL 论文 DRWeibo ACC | −0.0128 |
| 0.869 | 此前 Linux VM 单次 30-epoch 快速验证（非同配置） | +0.0122 |
| 0.866 | 交接文档所载 BiGCN ACC | +0.0152 |

只能表述为「本机官方配置复现得到 0.8812 ± 0.0045，与交接文档所载参考值存在差距」，**不能**据此宣称本机模型超过 BiGCN 或达到论文水平——BiGCN 尚未在本机运行，论文数字也未核实。可能的差异来源（均未通过调参去消除）：UWeibo 20k 用的是固定种子子集而非官方全量池随机抽样、gensim 4.3.2 / scikit-learn 1.0.2 等兼容性替代、Windows + PyTorch 1.12.0+cu113、模型初始化未设随机种子，以及论文数字本身待核实。全程未为涨点调整任何超参。

**并行效果修正**：此前预估剩 4 run 约 50–60 分钟，实测 2 worker 各跑 2 run 共 **90 分钟**；并行下单 run 从约 29 分钟升到约 45 分钟（两进程争抢 CPU/PCIe），但总墙钟仍短于串行 4×29≈116 分钟。GPU 利用率从单进程 17% 升到 82%，显存 2063MiB；提速来自并行，不是换设备或改模型。

复核命令：`python compute_final_metrics.py`（只读官方 JSON，生成 `RAGCL/logs/final_ragcl_drweibo_full_metrics.json`）。

## 6. 第二步：AD-GSCL 与 BiGCN 对比（进行中）

### 6.1 AD-GSCL 已完成（10 / 10 run）

- 固定 commit `819fe342…`，`main(sup).py` 官方源码未改；沿用 `run_adgscl_safe.py` 包装官方 `sort.py` 的批量删除。
- **统一口径**：AD-GSCL 官方默认是 `jieba + UWeiboV2`，与 RAGCL 输入不一致。为严格可比，改为 `--tokenize_mode naive --unsup_dataset UWeibo`，复用同一份 `w2v_DRWeiboV3_naive_20000_200.model`，使差异只来自模型/损失。此偏差已写入脚本注释与 manifest。
- 802 划分、10 run × 100 epoch、官方后 10 epoch 均值口径全部与 RAGCL 一致。
- 真实结果：`final10_mean_acc_avg = 0.8817 ± 0.0050`，macro-F1 `0.8811 ± 0.0051`；10 个 run 的 `exit_code` 全为 0，逐 run 耗时约 529–542 秒，证据见 `adgscl_manifest.json` 与 `AD-GSCL/logs/final_adgscl_drweibo_summary.json`。
- 与 RAGCL `0.8812 ± 0.0045` 几乎持平（差值 0.0005，远小于两者标准差），如实记录，不做「谁更好」的过度解读。

### 6.2 BiGCN 口径冲突与用户决策

官方 BiGCN（`5779f2cf…`）是 Weibo/TD-RvNN 格式、5000 维词袋、5-fold CV、200 epoch + EarlyStopping、torch 1.4 / PyG 1.3.2，与 RAGCL/AD-GSCL 的 DRWeiboV3 + word2vec + 802 + 10×100 完全不同。经确认，采用**统一到 DRWeiboV3** 的口径：

- `build_bigcn_drweibo.py`：复用 RAGCL `Main/sort.py` 的 802 划分文件名（不重新划分），在 **TRAIN-only** 文本上用 jieba 建 top-5000 词袋，输出 6037 个 `BiGCN/data/DRWeiboV3graph/*.npz`（压缩后 24.3 MB）、`vocab.json`、`split.json`。划分 train 4829 / val 1 / test 1208，标签 3185 / 2852。
- **必须声明的适配**：5000 维词袋是本次自建，**不是作者原始词表**；5-fold CV / EarlyStopping / 200 epoch 被 802 / 固定 100 epoch / 无早停替换；`num_workers 10→0`、`cuda:1→cuda:0`；DataLoader shuffle 顺序按统一协议。因此该数字**不能**与论文中 Weibo 上的 BiGCN 数值直接比较。
- `run_bigcn_drweibo.py`：用 `ast` 从官方 `BiGCN_Weibo.py` 抽取 `TDrumorGCN` / `BUrumorGCN` / `Net` 三个类原样执行（避开模块级训练循环），**模型结构与超参未改**：`Net(5000,64,64)`、lr 0.0005、weight_decay 1e-4、batch 16、BU conv lr÷5、nll_loss。评估改为 802 划分 + 10 run × 100 epoch + 后 10 epoch 均值。

### 6.3 工程问题与修复（均不改变模型数学）

1. **稠密特征缓存 7.6 GB**：6037 张图的 `x` 若全部缓存为 dense 会占约 7.6 GB。改为只缓存 scipy CSR（约 40 MB），`__getitem__` 时才把单图转 dense。
2. **官方 forward 在 CPU 分配大块临时张量**：官方 `th.zeros(n, 5000).to(device)` 每 batch 在 CPU 分配 20–32 MB 共 4 次。3–4 进程并发时物理内存（15.7 GB，空闲约 5.8 GB）被耗尽，报 `DefaultCPUAllocator: not enough memory` 或 `0xC0000005`。修复方式是在加载官方类时传入一个薄 `th` 代理，让 `zeros()` 直接在 GPU 上分配——`torch.zeros(n,m,device=cuda)` 与 `torch.zeros(n,m).to(cuda)` 数值完全相同，只改变临时张量所在设备。官方源码文件没有任何改动。
3. **并发数（实测后回退为单进程）**：4 / 3 进程触发 OOM；2 进程虽然把 GPU 利用率从单进程约 12% 提到约 85%，但系统 commit charge 达 30.3 GB（物理内存仅 15.7 GB），进程 RSS 从 2.1 GB 被换出到 1.3 GB，触发分页，单 epoch 从约 62 秒恶化到 >141 秒，总吞吐反而**低于串行**。因此最终改为 **1 worker 串行**：单进程约 2.1 GB RSS、空闲约 4.3 GB、约 62 秒/epoch。5000 维稠密节点特征使该工作负载受内存带宽限制，在本机 16 GB 内存下多进程不划算。
4. **断点续跑**：`resume_bigcn_runs.py` 每个 run 独立进程 + `--runs 1`，扫描 `BiGCN/Log/` 中 `epochs_recorded == 101` 的官方 JSON 判定完成，写 `bigcn_manifest.json`（命令、耗时、exit code、来源日志）与 `BiGCN/logs/final_bigcn_drweibo_summary.json`。2026-11-11 10:54 的 3-worker 批次在会话中断时整个进程树被杀，**没有任何 run 落盘**；2026-11-12 00:28 的 2-worker 批次因分页过慢被主动止损（同样无 run 落盘，不复用任何中途结果）；最终按 1 worker 串行重跑全部 10 run。中途失败/被杀的批次都不计入任何统计。
5. **运行环境**：后台 Bash 通道报 `/tmp` 缺失与 shim 权限错误（exit 125），改用前台启动后由超时机制自动转后台；`| tee` 会掩盖 Python 的真实退出码，因此驱动内部用 `subprocess.run` 直接重定向并记录 `returncode`。

### 6.3.1 2026-11-12 中断与断点恢复（如实记录）

- 00:55 启动的 1-worker 串行批次稳定跑到 10:44，完成 **run0–run6 共 7 个 run**，全部 `exit_code=0`、官方 JSON 中 `epochs_recorded=101`：后 10 epoch Test ACC 依次为 `0.8751 / 0.8708 / 0.8711 / 0.8727 / 0.8684 / 0.8671 / 0.8737`。这 7 份结果予以保留，**不重跑、不修改**。
- 10:54 **run7** 在 `evaluate()` 加载单图时抛出 `numpy.core._exceptions._ArrayMemoryError: Unable to allocate 2.94 MiB for an array with shape (154, 5000)`。原因是用户当时进行 GPU 测试，系统内存被占满；同一份代码此前已连续稳定完成 7 个 run，故判定为环境性 OOM，不是模型或脚本缺陷。**run8** 只打印到 `epoch 000`，随后用户主动停止整批进程。run7 / run8 均无完整 JSON，**不计入任何统计**，重跑时从 epoch 0 开始，不复用中间结果。
- 恢复前复核：`torch 1.12.0+cu113`、`cuda_available=True`、`device_count=1`、`cuda:0 = NVIDIA GeForce RTX 3080 Laptop GPU`（驱动 471.41，16 GB），GPU 矩阵乘通过；无残留训练进程，空闲物理内存 3.0 GB。
- 11:55 以 `--start 0 --end 10 --workers 1` 重启，`scan_completed()` 自动识别 `complete=[0..6]`、`targets=[7,8,9]`；驱动日志改用 `tee -a` 追加，保留 00:55 至今的完整时间线。
- **启动方式变化**：11:52 首次续跑沿用「前台启动 + 超时自动转后台」，本次在 120 秒超时被 `SIGTERM`，连同刚启动的 run7 子进程一起被杀（仅到 epoch 000）。改用工具托管的后台通道后启动成功。此前记录的「前台自动转后台可用」不再作为本机可靠路径。
- 13:24 / 14:43 / 16:02 依次完成 run7 / run8 / run9（均 `exit_code=0`，后 10 epoch ACC `0.8740 / 0.8695 / 0.8770`），16:02:40 写出 `BiGCN/logs/final_bigcn_drweibo_summary.json`：`status=complete`、`complete_runs=10`、`missing_runs=[]`。复核 10 份官方 JSON 全部 `epochs_recorded=101`。中途被杀/失败的 run7、run8 批次及 11:52 的短命批次均无完整 JSON，未计入任何统计。

### 6.4 三模型最终对比结果（2026-11-12 16:03 完成）

`compare_three_models.py` 只读真实日志生成 `three_model_comparison.json` 与 `COMPARISON-DRWEIBOV3.md`。三个模型 **各 10/10 完整 run**，全部为 DRWeiboV3 / 802 划分 / 10 run × 100 epoch / 官方后 10 epoch Test ACC 均值口径：

| 模型 | 角色 | Test ACC (mean ± SD) | 95% CI | macro-F1 |
|---|---|---|---|---|
| RAGCL (AAAI'24) | 主 baseline | **0.8812 ± 0.0045** | 0.8784–0.8840 | 0.8801 ± 0.0046 |
| AD-GSCL | 新一代对比 | **0.8817 ± 0.0050** | 0.8786–0.8848 | 0.8811 ± 0.0051 |
| BiGCN (AAAI'20) | 经典对照（适配版） | **0.8719 ± 0.0030** | 0.8701–0.8738 | 0.8717 ± 0.0029 |

BiGCN 逐 run 后 10 epoch ACC：`0.8751 / 0.8708 / 0.8711 / 0.8727 / 0.8684 / 0.8671 / 0.8737 / 0.8740 / 0.8695 / 0.8770`，独立重算均值 `0.871940`、总体 SD `0.002957`，与汇总脚本一致。

**读数说明（避免过度解读）**：

- RAGCL 与 AD-GSCL 差值仅 `0.0005`，远小于两者 run 间 SD（0.0045 / 0.0050），应视为**基本持平**，不宣称「AD-GSCL 更好」。
- BiGCN 比前两者低约 `0.0093–0.0098`（约 2–3 倍 SD），但它是**适配版**：自建词袋 + 802 划分 + 固定 100 epoch，输入特征与评估协议都不同，不能据此得出「BiGCN 模型弱于 RAGCL」的论文级结论，更不能与论文中 Weibo 上的 BiGCN 数值直接比较。
- 所有数字来自 `RAGCL/Log/*.json`、`AD-GSCL/Log/*.json`、`BiGCN/Log/*.json`，逐 run 证据见 `bigcn_manifest.json` / `adgscl_manifest.json` / `resume_manifest.json`；全程未为涨点调参，未填任何推测值。
- CI 统一按总体标准差（n，官方口径）的 SEM 计算：`mean ± 1.96 × SD/√10`。`final_ragcl_drweibo_full_metrics.json` 内另有一份按样本标准差（n−1）算的稍宽 CI `0.8782–0.8841`，仅为统计约定差异，均值与 SD 完全一致，不改变任何结论。

### 6.5 仍未开展 / 待确认

- 查新尚未开展；交接摘要里的时间感知评测、LLM 注入鲁棒性两个方向仍只是候选，不能当作「已确认无人做过」。
- 第二步产物（AD-GSCL / BiGCN 的运行脚本、官方日志、`*_manifest.json`、三模型对比表）**尚未提交 GitHub**；第一步代码已在 `miaopasi677/rumor-detection-ragcl` 私有仓库（`37da83f…`）。是否做第二次提交待负责人确认。
- 数据侧已确认：只有 DRWeiboV3 带完整时间戳（30 分钟 / 1 小时 / 6 小时 / 24 小时检查点可行，但 25% 源帖 30 分钟内无评论）；Weibo / Twitter15 / Twitter16 的 tf-idf 发布包无时间字段，不能默认做时间切分。
- 待客户/负责人确认：第二阶段范围与报价、署名与代码归属、是否要求严格单进程重跑 10 run、AD-GSCL 是否需要其预训练权重与半监督设置。
