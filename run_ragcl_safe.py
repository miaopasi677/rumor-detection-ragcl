"""Run official RAGCL main(sup).py without bulk-deleting prior split caches.

Main/sort.py calls Main/utils.py's dataset_makedirs(), which normally executes
shutil.rmtree() on a directory containing thousands of JSON files. This wrapper
replaces only that helper. Because sort.py uses a fixed random seed, every run
writes exactly the same 802-split JSON filenames; official write_post() opens
those files with "w" and overwrites them. We therefore only remove the three
processed data.pt caches (three individual deletes, below the safety threshold)
so TreeDataset.process() runs afresh for each run. No RAGCL source file is
modified, and no large split directory is archived or duplicated.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAGCL = ROOT / "RAGCL"
MAIN_SCRIPT = RAGCL / "Main" / "main(sup).py"
sys.path.insert(0, str(RAGCL))
# word2vec.py uses `from utils import word_tokenizer`, which relies on the
# Main directory being on sys.path when the script is executed directly.
sys.path.insert(0, str(RAGCL / "Main"))

import Main.sort  # noqa: E402


def safe_dataset_makedirs(dataset_path):
    root = Path(dataset_path)
    raw_paths = []
    removed = 0
    for split in ("train", "val", "test"):
        raw = root / split / "raw"
        processed = root / split / "processed"
        raw.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)
        cache = processed / "data.pt"
        if cache.exists():
            cache.unlink()
            removed += 1
        raw_paths.append(str(raw))
    print(
        f"[safe-dataset] removed {removed} processed caches; "
        "raw JSON files are overwritten by official sort_dataset().",
        flush=True,
    )
    return tuple(raw_paths)


Main.sort.dataset_makedirs = safe_dataset_makedirs

sys.argv = [str(MAIN_SCRIPT)] + sys.argv[1:]
runpy.run_path(str(MAIN_SCRIPT), run_name="__main__")
