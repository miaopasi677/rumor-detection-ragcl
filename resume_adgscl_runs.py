"""Crash-safe, batched AD-GSCL 10-run comparison on DRWeiboV3.

Each run is an independent `--runs 1` process using the official
AD-GSCL Main/main(sup).py through run_adgscl_safe.py. The official code,
802 split (sort.py random.seed(1234)), word2vec model and hyperparameters are
unchanged. Model initialisation is unseeded in the official code, so each
process gets an independent initialisation, just like consecutive runs inside
one process. Every completed run keeps its own official Log/*.json, and
adgscl_manifest.json records provenance so a killed batch can be resumed.

Comparison protocol (documented, not tuned):
  --dataset DRWeiboV3 --unsup_dataset UWeibo --tokenize_mode naive
  --runs 1 --epochs 100 --gpu 0
Using naive tokenisation + the exact same word2vec model as the RAGCL
reproduction makes the input features identical, so the only difference is
the model/loss (RAGCL graph augmentations vs AD-GSCL supervised contrastive
loss). AD-GSCL's official defaults are jieba + UWeiboV2; those are NOT used
here and must be noted in any report.
"""

import argparse
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADGSCL = ROOT / "AD-GSCL"
GPU_PYTHON = ROOT / "RAGCL" / ".venv-gpu" / "Scripts" / "python.exe"
WRAPPER = ROOT / "run_adgscl_safe.py"
LOG_DIR = ADGSCL / "Log"
CONSOLE_DIR = ADGSCL / "logs"
MANIFEST = ROOT / "adgscl_manifest.json"
SUMMARY = CONSOLE_DIR / "final_adgscl_drweibo_summary.json"

COMMON_ARGS = [
    "--dataset", "DRWeiboV3",
    "--unsup_dataset", "UWeibo",
    "--tokenize_mode", "naive",
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
            if len(accs) < 101:  # epoch 0 + 100 training epochs
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


def summarize(paths, out_path, target_runs):
    records = complete_records(paths)
    finals = [r["final10_mean_acc"] for r in records]
    bests = [r["best_epoch_acc"] for r in records]
    avg = sum(finals) / len(finals) if finals else None
    sd = math.sqrt(sum((x - avg) ** 2 for x in finals) / len(finals)) if finals else None
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "AD-GSCL main(sup)",
        "protocol": "DRWeiboV3 / 802 split / 10 runs x 100 epochs / naive w2v shared with RAGCL",
        "source_logs": [str(Path(p).name) for p in paths],
        "complete_runs": len(records),
        "per_run": records,
        "final10_mean_acc_avg": round(avg, 4) if avg is not None else None,
        "final10_mean_acc_population_sd": round(sd, 4) if sd is not None else None,
        "final10_mean_acc_min": round(min(finals), 4) if finals else None,
        "final10_mean_acc_max": round(max(finals), 4) if finals else None,
        "best_epoch_acc_avg": round(sum(bests) / len(bests), 4) if bests else None,
        "best_epoch_acc_max": round(max(bests), 4) if bests else None,
        "status": "complete" if len(records) >= target_runs else "partial",
        "target_runs": target_runs,
    }
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
                        help="Runs to execute in this batch; 0 means summarize only")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if MANIFEST.exists():
        manifest = read_json(MANIFEST)
    else:
        manifest = {
            "model": "AD-GSCL main(sup)",
            "protocol": "DRWeiboV3 / 802 / 10x100 / naive w2v (same as RAGCL)",
            "official_defaults_not_used": "jieba tokenisation and UWeiboV2 unlabelled graphs",
            "note": "Each run is an independent --runs 1 process; official source unchanged.",
            "completed": [],
        }
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    paths = [LOG_DIR / item["log_json"] for item in manifest["completed"]]
    done = len(complete_records(paths))
    remaining = max(0, args.target_runs - done)
    to_run = min(remaining, args.max_runs) if args.max_runs > 0 else 0
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] complete_runs={done} remaining={remaining} this_batch={to_run}", flush=True)

    for index in range(to_run):
        before = json_logs()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        console = CONSOLE_DIR / f"adgscl_run_{done + index:02d}_{stamp}.log"
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

    summarize(paths, SUMMARY, args.target_runs)


if __name__ == "__main__":
    main()
