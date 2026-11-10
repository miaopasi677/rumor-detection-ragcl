"""Recompute all final RAGCL DRWeiboV3 metrics from the official Log/*.json files.

This script does not train anything. It reads the source logs listed in
final_ragcl_drweibo_summary.json, recomputes per-run metrics from raw epoch
records, and writes a machine-readable report. Run it again to verify.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "RAGCL" / "Log"
DEFAULT_SUMMARY = ROOT / "RAGCL" / "logs" / "final_ragcl_drweibo_summary.json"
DEFAULT_OUTPUT = ROOT / "RAGCL" / "logs" / "final_ragcl_drweibo_full_metrics.json"

METRIC_KEYS = ("acc", "macro_precision", "macro_recall", "macro_f1", "f1_class0", "f1_class1")


def mean(values):
    return sum(values) / len(values)


def pop_sd(values):
    return statistics.pstdev(values)


def last10_mean(values):
    return mean(values[-10:])


def last10_biclass_mean(values):
    """Official JSON stores per-epoch [class0, class1]. Use the arithmetic mean."""
    return last10_mean([mean(pair) for pair in values[-10:]])


def describe(values):
    result = {
        "mean": round(mean(values), 6),
        "population_sd": round(pop_sd(values), 6),
        "sample_sd": round(statistics.stdev(values), 6) if len(values) > 1 else None,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }
    if len(values) > 1:
        sem = statistics.stdev(values) / math.sqrt(len(values))
        result["sem"] = round(sem, 6)
        result["ci95_normal"] = [
            round(mean(values) - 1.96 * sem, 6),
            round(mean(values) + 1.96 * sem, 6),
        ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("complete_runs") != 10:
        raise RuntimeError("Final summary is not a complete 10-run result")

    per_run = []
    for row in summary["per_run"]:
        source = row["source_log"]
        data = json.loads((LOG_DIR / source).read_text(encoding="utf-8"))
        run = next(r for r in data["record"] if r["run"] == row["run"])
        accs = run["test accs"]
        if len(accs) != 101:
            raise RuntimeError(f"{source} run {row['run']} has {len(accs)} epochs, expected 101")
        entry = {
            "source_log": source,
            "run": row["run"],
            "epochs_recorded": len(accs),
            "acc": round(last10_mean(accs), 6),
            "macro_precision": round(last10_biclass_mean(run["test precs"]), 6),
            "macro_recall": round(last10_biclass_mean(run["test recs"]), 6),
            "macro_f1": round(last10_biclass_mean(run["test f1s"]), 6),
            "f1_class0": round(last10_mean([v[0] for v in run["test f1s"][-10:]]), 6),
            "f1_class1": round(last10_mean([v[1] for v in run["test f1s"][-10:]]), 6),
            "best_epoch_acc": round(max(accs), 6),
            "official_mean_acc": run["mean acc"],
        }
        if abs(entry["acc"] - row["final10_mean_acc"]) > 0.00006:
            raise RuntimeError(f"ACC mismatch for {source} run {row['run']}")
        per_run.append(entry)

    groups = {
        "single_process_4runs": lambda s: s.startswith("2026-11-09 22-02-34"),
        "resume_2runs": lambda s: s in ("2026-11-10 18-48-17.json", "2026-11-10 19-16-23.json"),
        "shared_cache_parallel_4runs": lambda s: s in ("2026-11-10 19-58-57.json", "2026-11-10 19-59-04.json"),
    }

    report = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": "DRWeiboV3 (official Data.zip)",
        "config": "official RAGCL main(sup).py, 802 split, 10 runs x 100 epochs, --gpu 0",
        "metric_definition": {
            "primary": "mean of the last 10 epochs of test accs (official record_sup.py cal_mean=-10)",
            "sd": "population standard deviation across 10 runs (official record_sup.py convention)",
            "macro_precision_recall_f1": "arithmetic mean of the two per-class values per epoch, then mean over last 10 epochs (derived here; NOT the official 'test macro metric' field)",
            "best_epoch_acc": "max test acc over all 101 recorded epochs (secondary, not used as headline result)",
        },
        "source_logs": summary["source_logs"],
        "per_run": per_run,
        "pooled": {key: describe([r[key] for r in per_run]) for key in METRIC_KEYS},
        "pooled_best_epoch_acc": describe([r["best_epoch_acc"] for r in per_run]),
        "group_means_acc": {
            name: describe([r["acc"] for r in per_run if pred(r["source_log"])])
            for name, pred in groups.items()
        },
        "external_reference_values_not_produced_here": {
            "RAGCL_paper_DRWeibo_acc": 0.894,
            "previous_linux_vm_single_run_acc": 0.869,
            "BiGCN_acc_from_handover": 0.866,
            "note": "These come from the handover document, are unverified, and must not be presented as results of this machine.",
        },
        "provenance": summary["provenance"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print("ACC:", report["pooled"]["acc"]["mean"], "+/-", report["pooled"]["acc"]["population_sd"])
    print("macro-F1:", report["pooled"]["macro_f1"]["mean"], "+/-", report["pooled"]["macro_f1"]["population_sd"])
    print("best-epoch ACC:", report["pooled_best_epoch_acc"]["mean"])


if __name__ == "__main__":
    main()
