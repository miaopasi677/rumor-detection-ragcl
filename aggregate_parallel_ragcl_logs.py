"""Aggregate official RAGCL JSON logs from parallel worker copies."""

import argparse
import json
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values)


def population_sd(values):
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    logs = sorted((args.root).glob("worker_*/RAGCL/Log/*.json"))
    if not logs:
        raise RuntimeError("No worker JSON logs found")

    per_run = []
    worker_info = []
    for log_path in logs:
        with log_path.open("r", encoding="utf-8") as file:
            log = json.load(file)
        worker_info.append({"log_json": str(log_path), "declared_runs": log.get("runs"), "records": len(log["record"])})
        for run in log["record"]:
            accs = run["test accs"]
            per_run.append(
                {
                    "source": str(log_path),
                    "worker_run": run["run"],
                    "epoch_count": len(accs) - 1,
                    "final10_mean_acc": round(mean(accs[-10:]), 4),
                    "best_epoch_acc": round(max(accs), 4),
                    "final_epoch_acc": accs[-1],
                }
            )

    if len(per_run) != args.runs:
        raise RuntimeError(f"Expected {args.runs} runs, found {len(per_run)}")

    final_means = [item["final10_mean_acc"] for item in per_run]
    best_epochs = [item["best_epoch_acc"] for item in per_run]
    result = {
        "worker_logs": worker_info,
        "total_runs": len(per_run),
        "per_run": per_run,
        "final10_mean_acc_avg": round(mean(final_means), 4),
        "final10_mean_acc_population_sd": round(population_sd(final_means), 4),
        "final10_mean_acc_min": round(min(final_means), 4),
        "final10_mean_acc_max": round(max(final_means), 4),
        "best_epoch_acc_avg": round(mean(best_epochs), 4),
        "best_epoch_acc_max": round(max(best_epochs), 4),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
