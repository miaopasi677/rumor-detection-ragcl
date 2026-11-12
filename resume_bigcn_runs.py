"""Parallel, resumable driver for the BiGCN DRWeiboV3 10-run comparison.

Why this exists
---------------
Each run is launched as a separate process with --runs 1, exactly like
resume_ragcl_runs.py and resume_adgscl_runs.py, so:
  * an externally killed long-running shell only loses at most one run;
  * every run writes its own official-format JSON before exiting;
  * re-running this script skips runs whose JSON already has all 101 epochs.

Why workers defaults to 1 (MEASURED, do not "optimise" it back up):
  * 1 process: ~2.1 GB RSS, ~62 s/epoch, GPU ~12% (CPU/Python bound).
  * 3-4 processes: OOM (`DefaultCPUAllocator: not enough memory`, 0xC0000005).
  * 2 processes: GPU rose to ~85%, but commit charge hit 30.3 GB against
    15.7 GB physical RAM; RSS was paged out (2.1 GB -> 1.3 GB) and per-epoch
    time degraded from 62 s to >141 s. Total throughput was WORSE than serial.
  The 5000-dim dense node features make this workload memory-bandwidth bound,
  so parallel processes on a 16 GB host thrash the page file.

Nothing here changes the model, split, hyper-parameters or metric. All
results come from real run_bigcn_drweibo.py logs.

Typical use:
  python resume_bigcn_runs.py --start 1 --end 10 --workers 3
  python resume_bigcn_runs.py --summary-only
"""

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIGCN = ROOT / "BiGCN"
LOG_DIR = BIGCN / "Log"
CONSOLE_DIR = BIGCN / "logs"
PY = ROOT / "RAGCL" / ".venv-gpu" / "Scripts" / "python.exe"
TRAIN_SCRIPT = ROOT / "run_bigcn_drweibo.py"
MANIFEST = ROOT / "bigcn_manifest.json"
SUMMARY = CONSOLE_DIR / "final_bigcn_drweibo_summary.json"

EPOCHS = 100
EXPECTED_RECORDS = EPOCHS + 1  # epoch 0 is eval-only, then 100 trained epochs
TOTAL_RUNS = 10


def iso():
    return datetime.now().isoformat(timespec="seconds")


def scan_completed():
    """Map run index -> provenance, for runs with a complete official JSON."""
    done = {}
    for p in sorted(LOG_DIR.glob("bigcn_run*_*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for row in obj.get("record", []):
            if row.get("epochs_recorded", 0) < EXPECTED_RECORDS:
                continue  # smoke test or truncated run
            k = int(row["run"])
            if k not in done:
                done[k] = {
                    "run": k,
                    "log_json": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "final10_mean_acc": row["final10_mean_acc"],
                    "final10_mean_macro_f1": row["final10_mean_macro_f1"],
                    "best_epoch_acc": row["best_epoch_acc"],
                    "elapsed_seconds": row["elapsed_seconds"],
                }
    return done


def write_manifest(entries):
    MANIFEST.write_text(json.dumps({
        "model": "BiGCN (official Net, DRWeiboV3 adapted)",
        "protocol": "802 split / 10 runs x 100 epochs / last-10 mean",
        "driver": "resume_bigcn_runs.py",
        "updated_at": iso(),
        "runs": sorted(entries.values(), key=lambda e: e["run"]),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary(done):
    rows = [done[k] for k in sorted(done) if k < TOTAL_RUNS]
    finals = [r["final10_mean_acc"] for r in rows]
    f1s = [r["final10_mean_macro_f1"] for r in rows]
    n = len(finals)
    avg = sum(finals) / n if n else 0.0
    sd = math.sqrt(sum((x - avg) ** 2 for x in finals) / n) if n else 0.0
    sem = sd / math.sqrt(n) if n else 0.0
    f1avg = sum(f1s) / n if n else 0.0
    f1sd = math.sqrt(sum((x - f1avg) ** 2 for x in f1s) / n) if n else 0.0
    missing = [k for k in range(TOTAL_RUNS) if k not in done]
    out = {
        "model": "BiGCN (official Net, DRWeiboV3 adapted)",
        "protocol": "802 split / 10 runs x 100 epochs / last-10 test-ACC mean",
        "status": "complete" if not missing and n == TOTAL_RUNS else "partial",
        "complete_runs": n,
        "missing_runs": missing,
        "final10_mean_acc_avg": round(avg, 4),
        "final10_mean_acc_population_sd": round(sd, 4),
        "final10_mean_acc_sem": round(sem, 4),
        "final10_mean_acc_ci95_normal": [round(avg - 1.96 * sem, 4),
                                         round(avg + 1.96 * sem, 4)] if n else None,
        "final10_mean_macro_f1_avg": round(f1avg, 4),
        "final10_mean_macro_f1_population_sd": round(f1sd, 4),
        "per_run": rows,
        "generated_at": iso(),
        "note": ("BiGCN features are a jieba TRAIN-only top-5000 bag-of-words "
                 "built from DRWeiboV3, NOT the authors' Weibo vocabulary; "
                 "5-fold CV and EarlyStopping are replaced by the unified "
                 "802 / fixed-100-epoch protocol. See run_bigcn_drweibo.py."),
    }
    CONSOLE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    return out


def run_one(k, epochs, gpu, stagger=0):
    # Stagger CUDA-context / CSR-cache construction so 3 processes do not
    # allocate their peak working set in the same second.
    if stagger:
        time.sleep(5.0 * stagger)
    tag = f"run{k:02d}"
    cmd = [str(PY), "-u", str(TRAIN_SCRIPT),
           "--runs", "1", "--epochs", str(epochs),
           "--tag", tag, "--run-offset", str(k), "--gpu", str(gpu)]
    console = CONSOLE_DIR / f"bigcn_{tag}.log"
    entry = {
        "run": k,
        "command": " ".join(f'"{c}"' if " " in c else c for c in cmd),
        "console_log": str(console.relative_to(ROOT)).replace("\\", "/"),
        "started_at": iso(),
    }
    t0 = time.time()
    with console.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f,
                              stderr=subprocess.STDOUT)
    entry.update({
        "finished_at": iso(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "exit_code": proc.returncode,
    })
    if proc.returncode == 0:
        done = scan_completed()
        if k in done:
            entry.update({kk: vv for kk, vv in done[k].items() if kk != "run"})
        else:
            entry["error"] = "exit 0 but no complete JSON found"
    else:
        entry["error"] = f"nonzero exit {proc.returncode}; see console_log"
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=TOTAL_RUNS, help="exclusive")
    ap.add_argument("--workers", type=int, default=1,
                    help="1 is the measured optimum on this 16 GB host; "
                         "2+ processes page-thrash and are slower than serial")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLE_DIR.mkdir(parents=True, exist_ok=True)
    done = scan_completed()

    if args.summary_only:
        out = write_summary(done)
        print(json.dumps({k: v for k, v in out.items() if k != "per_run"},
                         ensure_ascii=False, indent=2))
        return

    targets = [k for k in range(args.start, args.end) if k not in done]
    print(f"[{iso()}] complete={sorted(done)} targets={targets} "
          f"workers={args.workers}", flush=True)
    if not targets:
        write_summary(done)
        return

    entries = {}
    if MANIFEST.exists():
        try:
            for e in json.loads(MANIFEST.read_text(encoding="utf-8")).get("runs", []):
                entries[int(e["run"])] = e
        except (OSError, ValueError):
            pass

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, k, args.epochs, args.gpu, i % args.workers): k
                   for i, k in enumerate(targets)}
        for fut in as_completed(futures):
            k = futures[fut]
            try:
                e = fut.result()
            except Exception as exc:  # keep the other runs alive
                e = {"run": k, "error": repr(exc), "finished_at": iso()}
            entries[k] = e
            write_manifest(entries)
            print(f"[{iso()}] run {k} exit={e.get('exit_code')} "
                  f"acc={e.get('final10_mean_acc')} "
                  f"elapsed={e.get('elapsed_seconds')}s "
                  f"error={e.get('error', '-')}", flush=True)

    done = scan_completed()
    out = write_summary(done)
    print(f"[{iso()}] SUMMARY status={out['status']} n={out['complete_runs']} "
          f"mean={out['final10_mean_acc_avg']} sd={out['final10_mean_acc_population_sd']}",
          flush=True)


if __name__ == "__main__":
    main()
