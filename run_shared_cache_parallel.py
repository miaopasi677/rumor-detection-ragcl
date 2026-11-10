"""Finish the remaining RAGCL runs with two GPU workers sharing one graph cache.

Both workers use the same official RAGCL source, the same fixed 802 split and
one shared processed graph cache (run_ragcl_shared_cache.py never deletes the
three data.pt files). Each worker gets its own RAGCL_RAW_ROOT so concurrent
sort_dataset() writes never touch the same file. Workers start a few seconds
apart so their official Log/<timestamp>.json names cannot collide.

--collect-only re-aggregates official JSON logs from workers whose orchestrator
was killed, so completed runs are never lost.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from resume_ragcl_runs import (  # noqa: E402
    GPU_PYTHON, LOG_DIR, MANIFEST, SUMMARY, complete_records,
    json_logs, read_json, summarize,
)

WRAPPER = ROOT / "run_ragcl_shared_cache.py"
LOG_ROOT = ROOT / "parallel_shared" / "logs"


def append_manifest(manifest, new_logs, extra=None):
    for path in new_logs:
        entry = {
            "log_json": path.name,
            "console_log": "parallel_shared/logs/",
            "command": "shared-cache parallel workers",
            "mode": "shared processed graph cache, per-worker raw split directory",
        }
        if extra:
            entry.update(extra)
        manifest["completed"].append(entry)
    manifest["shared_cache_parallel"] = {
        "cache_source": "RAGCL/Data/DRWeiboV3/dataset/{train,val,test}/processed/data.pt",
        "note": "Fixed 802 split + fixed word2vec; deterministic caches skip TreeDataset.process().",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finish(known_logs, new_logs, target_runs, provenance):
    all_paths = known_logs + new_logs
    summarize(all_paths, SUMMARY, extra={
        "target_runs": target_runs,
        "status": "complete" if len(complete_records(all_paths)) >= target_runs else "partial",
        "provenance": provenance,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stagger-seconds", type=int, default=3)
    parser.add_argument("--target-runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    manifest = read_json(MANIFEST)
    known_logs = [LOG_DIR / manifest["baseline_log"]] + [
        LOG_DIR / item["log_json"] for item in manifest["completed"]
    ]
    done = len(complete_records(known_logs))
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] complete_runs={done}", flush=True)

    if args.collect_only:
        known_names = {p.name for p in known_logs}
        new_logs = [
            p for p in sorted(json_logs())
            if p.name not in known_names and complete_records([p])
        ]
        print(f"collect-only new logs: {[p.name for p in new_logs]}", flush=True)
        append_manifest(manifest, new_logs)
        finish(known_logs, new_logs, args.target_runs,
               "single process + resume processes + shared-cache parallel workers (collected after orchestrator loss)")
        return

    remaining = max(0, args.target_runs - done)
    print(f"remaining={remaining} workers={args.workers}", flush=True)
    if remaining == 0:
        finish(known_logs, [], args.target_runs, "all runs already complete")
        return

    counts = [remaining // args.workers + (i < remaining % args.workers) for i in range(args.workers)]
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    before = json_logs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    workers = []
    for index, count in enumerate(counts):
        if count == 0:
            continue
        console = LOG_ROOT / f"worker_{index:02d}_{stamp}.log"
        raw_root = str(LOG_ROOT.parent / "raw" / f"worker_{index:02d}")
        command = [
            str(GPU_PYTHON), "-u", str(WRAPPER),
            "--dataset", "DRWeiboV3", "--unsup_dataset", "UWeibo",
            "--gpu", "0", "--runs", str(count), "--epochs", str(args.epochs),
        ]
        env = os.environ.copy()
        env["RAGCL_RAW_ROOT"] = raw_root
        stream = console.open("w", encoding="utf-8", errors="replace")
        stream.write("command: " + " ".join(command) + "\n")
        stream.write("RAGCL_RAW_ROOT=" + raw_root + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=stream,
                                   stderr=subprocess.STDOUT, env=env)
        workers.append({
            "index": index, "runs": count, "command": " ".join(command),
            "console": console.name, "raw_root": raw_root,
            "process": process, "stream": stream,
            "started": datetime.now().isoformat(timespec="seconds"), "t0": time.time(),
            "exit_code": None, "elapsed_seconds": None,
        })
        print(f"worker {index:02d}: pid={process.pid} runs={count} console={console.name}", flush=True)
        if args.stagger_seconds > 0 and index < len(counts) - 1 and any(c > 0 for c in counts[index + 1:]):
            time.sleep(args.stagger_seconds)

    for worker in workers:
        code = worker["process"].wait()
        worker["stream"].close()
        worker["exit_code"] = code
        worker["elapsed_seconds"] = round(time.time() - worker["t0"], 1)
        print(f"worker {worker['index']:02d}: exit={code} elapsed={worker['elapsed_seconds']}s", flush=True)

    new_logs = [p for p in sorted(json_logs() - before) if complete_records([p])]
    print(f"new official JSON logs: {[p.name for p in new_logs]}", flush=True)

    status = {
        "started": stamp,
        "complete_runs_before": done,
        "remaining": remaining,
        "workers": [
            {k: v for k, v in w.items() if k not in ("process", "stream", "t0")}
            for w in workers
        ],
        "new_logs": [p.name for p in new_logs],
        "all_succeeded": all(w["exit_code"] == 0 for w in workers),
    }
    (LOG_ROOT.parent / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    append_manifest(manifest, new_logs, extra={
        "exit_codes": [w["exit_code"] for w in workers],
        "stagger_seconds": args.stagger_seconds,
    })
    finish(known_logs, new_logs, args.target_runs,
           "single process + resume processes + 2 shared-cache parallel workers")

    if not status["all_succeeded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
