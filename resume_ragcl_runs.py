"""Crash-safe, batched resume of the official RAGCL 10-run reproduction.

The 2026-11-09 22:02 single-process run was externally terminated after four
complete runs. Official main(sup).py only appends a run to its JSON after all
100 epochs finish, so the interrupted fifth run produced no JSON record.

This driver runs the remaining runs as independent `--runs 1` processes. The
official code, arguments, 802 split (sort.py's random.seed(1234)) and word2vec
model are unchanged. Model initialisation is unseeded in the official code, so
each process gets its own independent initialisation, just as consecutive runs
inside one process do. Every completed run keeps its own official Log/*.json,
and resume_manifest.json records provenance so a killed batch can be resumed.
"""

import argparse
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAGCL = ROOT / "RAGCL"
GPU_PYTHON = RAGCL / ".venv-gpu" / "Scripts" / "python.exe"
WRAPPER = ROOT / "run_ragcl_safe.py"
LOG_DIR = RAGCL / "Log"
CONSOLE_DIR = RAGCL / "logs"
BASELINE_LOG = LOG_DIR / "2026-11-09 22-02-34.json"
MANIFEST = ROOT / "resume_manifest.json"
SUMMARY = CONSOLE_DIR / "final_ragcl_drweibo_summary.json"

COMMON_ARGS = [
    "--dataset", "DRWeiboV3",
    "--unsup_dataset", "UWeibo",
    "--gpu", "0",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def complete_records(paths):
    records = []
    for path in paths:
        log = read_json(path)
        for run in log.get("record", []):
            accs = run.get("test accs", [])
            # epoch 0 plus 100 training epochs; shorter means an interrupted run
            if len(accs) < 101:
                continue
            final10 = sum(accs[-10:]) / 10
            records.append({
                "source_log": str(Path(path).name),
                "run": run["run"],
                "epochs_recorded": len(accs),
                "final10_mean_acc": round(final10, 4),
                "best_epoch_acc": round(max(accs), 4),
                "official_mean_acc": run.get("mean acc"),
            })
    return records


def summarize(paths, out_path, extra=None):
    records = complete_records(paths)
    finals = [r["final10_mean_acc"] for r in records]
    bests = [r["best_epoch_acc"] for r in records]
    avg = sum(finals) / len(finals) if finals else None
    sd = math.sqrt(sum((x - avg) ** 2 for x in finals) / len(finals)) if finals else None
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_logs": [str(Path(p).name) for p in paths],
        "complete_runs": len(records),
        "per_run": records,
        "final10_mean_acc_avg": round(avg, 4) if avg is not None else None,
        "final10_mean_acc_population_sd": round(sd, 4) if sd is not None else None,
        "final10_mean_acc_max": round(max(finals), 4) if finals else None,
        "best_epoch_acc_avg": round(sum(bests) / len(bests), 4) if bests else None,
        "best_epoch_acc_max": round(max(bests), 4) if bests else None,
    }
    if extra:
        summary.update(extra)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return summary


def json_logs():
    return set(LOG_DIR.glob("*.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-runs", type=int, default=10)
    parser.add_argument("--max-runs", type=int, default=2,
                        help="Maximum number of runs to execute in this batch; 0 means summarize only")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if MANIFEST.exists():
        manifest = read_json(MANIFEST)
    else:
        manifest = {
            "baseline_log": str(BASELINE_LOG.name),
            "interrupted_process": "2026-11-09 22-02-34 (terminated 2026-11-10 00:05, 4 complete runs)",
            "note": "Remaining runs are independent --runs 1 processes with identical official settings.",
            "completed": [],
        }
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    paths = [LOG_DIR / manifest["baseline_log"]] + [LOG_DIR / item["log_json"] for item in manifest["completed"]]
    done = len(complete_records(paths))
    remaining = max(0, args.target_runs - done)
    to_run = min(remaining, args.max_runs) if args.max_runs > 0 else 0
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] complete_runs={done} remaining={remaining} this_batch={to_run}", flush=True)

    for index in range(to_run):
        before = json_logs()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        console = CONSOLE_DIR / f"resume_run_{done + index:02d}_{stamp}.log"
        command = [str(GPU_PYTHON), "-u", str(WRAPPER), *COMMON_ARGS,
                   "--runs", "1", "--epochs", str(args.epochs)]
        started = datetime.now().isoformat(timespec="seconds")
        t0 = time.time()
        with console.open("w", encoding="utf-8", errors="replace") as stream:
            stream.write("command: " + " ".join(command) + "\n")
            stream.flush()
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=str(ROOT))
        elapsed = round(time.time() - t0, 1)
        new_logs = sorted(json_logs() - before)
        print(f"run {done + index}: exit={proc.returncode} elapsed={elapsed}s new_logs={[p.name for p in new_logs]}", flush=True)
        if proc.returncode != 0 or len(new_logs) != 1:
            print("Stopping batch: run did not produce exactly one complete official JSON.", flush=True)
            break
        manifest["completed"].append({
            "log_json": new_logs[0].name,
            "console_log": console.name,
            "command": " ".join(command),
            "started": started,
            "elapsed_seconds": elapsed,
            "exit_code": proc.returncode,
        })
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(new_logs[0])

    summarize(paths, SUMMARY, extra={
        "target_runs": args.target_runs,
        "status": "complete" if len(complete_records(paths)) >= args.target_runs else "partial",
        "provenance": "4 runs from the 2026-11-09 22:02 single process + independent --runs 1 resume processes",
    })


if __name__ == "__main__":
    main()
