"""Launch the 10-run RAGCL reproduction across isolated worker copies.

Each worker uses the same official Main/*.py source, hard-linked DRWeiboV3
source JSONs, and the same trained word2vec model. Workers get separate
Data/DRWeiboV3/dataset and Log directories so sort_dataset() cannot race.
The default three workers run 4/3/3 official runs, for ten runs in total.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAGCL = ROOT / "RAGCL"
GPU_PYTHON = RAGCL / ".venv-gpu" / "Scripts" / "python.exe"
SOURCE = RAGCL / "Data" / "DRWeiboV3" / "source"
MODEL = RAGCL / "Model" / "w2v_DRWeiboV3_naive_20000_200.model"
WRAPPER = ROOT / "run_ragcl_safe.py"
RUNS = 10
EPOCHS = 100


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        import os

        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def setup_worker(worker, run_count):
    root = ROOT / "parallel_runs" / f"worker_{worker:02d}"
    worker_ragcl = root / "RAGCL"
    (worker_ragcl / "Log").mkdir(parents=True, exist_ok=True)
    shutil.copytree(RAGCL / "Main", worker_ragcl / "Main", dirs_exist_ok=True)
    shutil.copy2(WRAPPER, root / WRAPPER.name)
    link_or_copy(MODEL, worker_ragcl / "Model" / MODEL.name)
    for source_file in SOURCE.glob("*.json"):
        link_or_copy(source_file, worker_ragcl / "Data" / "DRWeiboV3" / "source" / source_file.name)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    counts = [args.runs // args.workers + (i < args.runs % args.workers) for i in range(args.workers)]
    output_root = ROOT / "parallel_runs"
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    processes = []
    started = time.monotonic()
    for worker, run_count in enumerate(counts):
        if run_count == 0:
            continue
        root = setup_worker(worker, run_count)
        command = [
            str(GPU_PYTHON),
            "-u",
            str(root / WRAPPER.name),
            "--dataset",
            "DRWeiboV3",
            "--unsup_dataset",
            "UWeibo",
            "--runs",
            str(run_count),
            "--epochs",
            str(args.epochs),
            "--gpu",
            "0",
        ]
        stdout = (logs / f"worker_{worker:02d}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=subprocess.STDOUT)
        processes.append((worker, run_count, root, process, stdout))
        print(f"worker {worker:02d}: pid={process.pid}, runs={run_count}, root={root}", flush=True)

    results = []
    for worker, run_count, root, process, stdout in processes:
        code = process.wait()
        stdout.close()
        results.append({"worker": worker, "runs": run_count, "root": str(root), "exit_status": code})
        print(f"worker {worker:02d}: finished, exit_status={code}", flush=True)

    summary = {
        "total_runs": args.runs,
        "epochs": args.epochs,
        "workers": results,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "all_succeeded": all(item["exit_status"] == 0 for item in results),
    }
    (output_root / "status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["all_succeeded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
