"""Run official RAGCL main(sup).py while reusing a fixed processed graph cache.

Only Main.sort.dataset_makedirs is replaced; RAGCL source is untouched.
Unlike run_ragcl_safe.py, the three processed/data.pt caches are NOT deleted.
The 802 split is fixed by sort.py's random.seed(1234), and word2vec,
centrality and graph construction are deterministic, so reusing the cache
produces identical tensors while avoiding the concurrent process() memory
spike that previously killed parallel workers.

If any cache is missing, this wrapper fails fast instead of rebuilding graphs
in parallel (which would risk the earlier out-of-memory failure).
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAGCL = ROOT / "RAGCL"
# Each parallel worker writes split JSONs into its own directory, so two
# sort_dataset() calls never touch the same file. Processed caches stay shared
# because that is what TreeDataset actually reads.
WORKER_RAW_ROOT = os.environ.get("RAGCL_RAW_ROOT")
MAIN_SCRIPT = RAGCL / "Main" / "main(sup).py"
sys.path.insert(0, str(RAGCL))
# word2vec.py imports `from utils import word_tokenizer` when run directly.
sys.path.insert(0, str(RAGCL / "Main"))

import Main.sort  # noqa: E402


def shared_cache_dataset_makedirs(dataset_path):
    root = Path(dataset_path)
    raw_paths = []
    missing = []
    for split in ("train", "val", "test"):
        if WORKER_RAW_ROOT:
            raw = Path(WORKER_RAW_ROOT) / split / "raw"
        else:
            raw = root / split / "raw"
        processed = root / split / "processed"
        raw.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        cache = processed / "data.pt"
        if not cache.exists():
            missing.append(str(cache))
        raw_paths.append(str(raw))
    if missing:
        raise RuntimeError("Shared graph cache missing: " + ", ".join(missing))
    print(f"[shared-cache] reusing 3 processed data.pt caches; raw root={WORKER_RAW_ROOT or root}", flush=True)
    return tuple(raw_paths)


Main.sort.dataset_makedirs = shared_cache_dataset_makedirs

sys.argv = [str(MAIN_SCRIPT)] + sys.argv[1:]
runpy.run_path(str(MAIN_SCRIPT), run_name="__main__")
