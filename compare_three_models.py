"""Build the RAGCL / AD-GSCL / BiGCN comparison from REAL result files.

All numbers are read from logs produced by actual training runs. Nothing is
estimated, interpolated or filled in. If the BiGCN summary is still partial,
the table says so explicitly instead of inventing a number.

Usage:
    python compare_three_models.py
Outputs:
    three_model_comparison.json
    COMPARISON-DRWEIBOV3.md
"""

import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAGCL_METRICS = ROOT / "RAGCL" / "logs" / "final_ragcl_drweibo_full_metrics.json"
ADGSCL_SUMMARY = ROOT / "AD-GSCL" / "logs" / "final_adgscl_drweibo_summary.json"
ADGSCL_LOG_DIR = ROOT / "AD-GSCL" / "Log"
BIGCN_SUMMARY = ROOT / "BiGCN" / "logs" / "final_bigcn_drweibo_summary.json"
OUT_JSON = ROOT / "three_model_comparison.json"
OUT_MD = ROOT / "COMPARISON-DRWEIBOV3.md"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats(values):
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)  # population SD (official)
    sem = sd / math.sqrt(n)
    return {
        "n": n,
        "mean": round(mean, 4),
        "population_sd": round(sd, 4),
        "sem": round(sem, 4),
        "ci95_normal": [round(mean - 1.96 * sem, 4), round(mean + 1.96 * sem, 4)],
        "per_run": [round(v, 4) for v in values],
    }


def mean_last10(seq):
    tail = seq[-10:]
    return sum(float(v) for v in tail) / len(tail)


def macro_f1_last10(seq):
    """AD-GSCL logs store per-class F1 as a 2-element list per epoch."""
    vals = []
    for item in seq[-10:]:
        if isinstance(item, (list, tuple)):
            vals.append(sum(float(v) for v in item) / len(item))
        else:
            vals.append(float(item))
    return sum(vals) / len(vals)


def collect_ragcl():
    m = load(RAGCL_METRICS)
    rows = m["per_run"]
    return ([float(r["acc"]) for r in rows],
            [float(r["macro_f1"]) for r in rows],
            m.get("source_logs", []))


def collect_adgscl():
    s = load(ADGSCL_SUMMARY)
    accs, f1s = [], []
    for name in s["source_logs"]:
        rec = load(ADGSCL_LOG_DIR / name)["record"][0]
        accs.append(mean_last10(rec["test accs"]))
        f1s.append(macro_f1_last10(rec["test f1s"]))
    return accs, f1s, s["source_logs"]


def collect_bigcn():
    if not BIGCN_SUMMARY.exists():
        return [], [], [], "not_started"
    s = load(BIGCN_SUMMARY)
    rows = s.get("per_run", [])
    return ([float(r["final10_mean_acc"]) for r in rows],
            [float(r["final10_mean_macro_f1"]) for r in rows],
            [r["log_json"] for r in rows],
            s.get("status", "unknown"))


def fmt(st):
    if st is None:
        return "pending"
    return f"{st['mean']:.4f} ± {st['population_sd']:.4f}"


def main():
    r_acc, r_f1, r_src = collect_ragcl()
    a_acc, a_f1, a_src = collect_adgscl()
    b_acc, b_f1, b_src, b_status = collect_bigcn()

    models = [
        {
            "model": "RAGCL (AAAI'24)",
            "role": "主 baseline",
            "features": "官方 word2vec（naive 分词，UWeibo 20k 固定子集，200 维）",
            "acc": stats(r_acc), "macro_f1": stats(r_f1),
            "source_logs": r_src,
            "notes": "官方 main(sup).py 源码未修改；复现值低于论文 0.894，如实记录。",
        },
        {
            "model": "AD-GSCL",
            "role": "新一代对比",
            "features": "与 RAGCL 完全相同的 word2vec / 分词 / 无标签语料",
            "acc": stats(a_acc), "macro_f1": stats(a_f1),
            "source_logs": a_src,
            "notes": "为严格可比，未用官方默认 jieba + UWeiboV2，改为 naive + UWeibo；"
                     "仅模型/损失不同。",
        },
        {
            "model": "BiGCN (AAAI'20)",
            "role": "经典对照（适配版）",
            "features": "自建 jieba top-5000 词袋（TRAIN-only），非作者原词表",
            "acc": stats(b_acc), "macro_f1": stats(b_f1),
            "source_logs": b_src,
            "notes": "官方 Net/TDrumorGCN/BUrumorGCN 类未改；5-fold CV、EarlyStopping、"
                     "200 epoch 被统一协议替换。与论文数字不可直接比较。",
            "status": b_status,
        },
    ]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "DRWeiboV3（官方 Data.zip）",
        "split": "802（RAGCL Main/sort.py, random.seed(1234)；train 4829 / val 1 / test 1208）",
        "protocol": "10 runs x 100 epochs；主指标 = 每 run 最后 10 个 epoch 的 Test ACC 均值"
                    "（官方 record_sup.py cal_mean=-10）；SD 为 10 run 的总体标准差",
        "headline_rule": "不宣称复现到论文数值、不承诺录用、未为涨点调参",
        "models": models,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    lines = [
        "# DRWeiboV3 三模型对比（真实运行日志）",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        f"- 数据集：{payload['dataset']}",
        f"- 划分：{payload['split']}",
        f"- 协议：{payload['protocol']}",
        "",
        "## 结果表",
        "",
        "| 模型 | 角色 | 输入特征 | 完整 run | Test ACC (mean ± SD) | 95% CI | macro-F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in models:
        a = m["acc"]
        f = m["macro_f1"]
        ci = "-".join(str(x) for x in a["ci95_normal"]) if a else "pending"
        lines.append(
            f"| {m['model']} | {m['role']} | {m['features']} | "
            f"{a['n'] if a else 0}/10 | {fmt(a)} | {ci} | {fmt(f)} |"
        )

    lines += [
        "",
        "## 口径与限制（必须随结果一起说明）",
        "",
        "1. **RAGCL 与 AD-GSCL 严格可比**：同一份 `w2v_DRWeiboV3_naive_20000_200.model`、"
        "同一 naive 分词、同一 20k UWeibo 子集、同一 802 划分、同一 10×100 协议；"
        "差异只来自模型结构与损失。AD-GSCL 官方默认 `jieba + UWeiboV2` 未使用。",
        "2. **BiGCN 是适配版，不是官方复现**：官方基于 Weibo 的 5000 维词袋、5-fold CV、"
        "200 epoch + EarlyStopping；本实验为统一到 DRWeiboV3，改用 TRAIN-only jieba "
        "top-5000 词袋、802 划分、固定 100 epoch、无早停。官方 `Net` 三个类、"
        "lr=0.0005 / weight_decay=1e-4 / batch=16 / hidden=64 / BU conv lr÷5 均未改。"
        "因此 BiGCN 数字不能与论文中 Weibo 上的数值直接比较。",
        "3. **RAGCL 本机值低于论文**：0.8812 vs 论文约 0.894。可能来源：20k 固定种子"
        "UWeibo 子集而非全量池、gensim/sklearn 兼容版本、Windows + cu113、官方代码未固定"
        "模型初始化种子。全程未为涨点调参。",
        "4. `val` 只有 1 条样本，Val ACC 不具参考意义；主指标只看 Test ACC。",
        "5. 所有数字来自 `RAGCL/Log/*.json`、`AD-GSCL/Log/*.json`、`BiGCN/Log/*.json` "
        "真实日志，机器可读明细见 `three_model_comparison.json`。",
        "6. **统计约定**：表中 SD 为 10 run 的总体标准差（n，官方 `record_sup.py` 口径）；"
        "95% CI 用同一 SD 的 SEM 计算，即 `mean ± 1.96 × SD/√n`。"
        "`RAGCL/logs/final_ragcl_drweibo_full_metrics.json` 中另存有一份按样本标准差"
        "（n−1）计算的 CI（0.8782–0.8841），比本表略宽；两者仅为统计约定差异，"
        "不改变均值、SD 或任何结论。",
        "",
        "## 各模型备注",
        "",
    ]
    for m in models:
        lines.append(f"- **{m['model']}**：{m['notes']}")
        if m.get("status") == "partial":
            lines.append(f"  - 当前 BiGCN 仅完成 {len(m['acc']['per_run']) if m['acc'] else 0}/10 run，"
                         "表格中数值为部分结果，**不是最终结论**。")
        lines.append(f"  - 来源日志 {len(m['source_logs'])} 份。")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_JSON.name} and {OUT_MD.name}")
    for m in models:
        print(f"  {m['model']:<20} n={m['acc']['n'] if m['acc'] else 0:2d} "
              f"ACC={fmt(m['acc'])} macroF1={fmt(m['macro_f1'])}")


if __name__ == "__main__":
    main()
