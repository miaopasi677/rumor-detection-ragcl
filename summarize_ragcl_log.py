"""Summarize an official RAGCL supervised-training JSON log.

It reports both:
1. the mean of each run's final 10 test accuracies (the paper's protocol), and
2. the best single test-epoch accuracy in each run (diagnostic only).
"""

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
    parser.add_argument("log_json", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    with args.log_json.open("r", encoding="utf-8") as file:
        log = json.load(file)

    per_run = []
    for run in log["record"]:
        accs = run["test accs"]
        final10 = round(mean(accs[-10:]), 4)
        best = round(max(accs), 4)
        per_run.append({"run": run["run"], "final10_mean_acc": final10, "best_epoch_acc": best})

    final_means = [item["final10_mean_acc"] for item in per_run]
    best_epochs = [item["best_epoch_acc"] for item in per_run]
    summary = {
        "log_json": str(args.log_json.resolve()),
        "dataset": log.get("dataset"),
        "epochs": log.get("epochs"),
        "runs": log.get("runs"),
        "per_run": per_run,
        "final10_mean_acc_avg": round(mean(final_means), 4),
        "final10_mean_acc_population_sd": round(population_sd(final_means), 4),
        "final10_mean_acc_max": round(max(final_means), 4),
        "best_epoch_acc_avg": round(mean(best_epochs), 4),
        "best_epoch_acc_max": round(max(best_epochs), 4),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
