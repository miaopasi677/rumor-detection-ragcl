"""Train the word2vec model using RAGCL's unmodified official helper functions.

The CPU torch build is sufficient here: word2vec.py imports torch but does not
use CUDA. The saved model name matches Main/main(sup).py's expected filename
for --dataset DRWeiboV3 --unsup_train_size 20000 --vector_size 200.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAGCL = ROOT / "RAGCL"
MAIN = RAGCL / "Main"
sys.path.insert(0, str(MAIN))

# word2vec.py uses `from utils import ...`; RAGCL/Main must be first on sys.path.
import word2vec  # noqa: E402


def main():
    label_source_path = RAGCL / "Data" / "DRWeiboV3" / "source"
    unlabel_dataset_path = RAGCL / "Data" / "UWeibo" / "dataset"
    output_path = RAGCL / "Model" / "w2v_DRWeiboV3_naive_20000_200.model"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    print("Collecting official DRWeiboV3 + 20k UWeibo sentences...", flush=True)
    sentences = word2vec.collect_sentences(
        str(label_source_path),
        str(unlabel_dataset_path),
        20000,
        lang="ch",
        tokenize_mode="naive",
    )
    print(f"Sentence count: {len(sentences)}", flush=True)

    print("Training word2vec with official settings (30 epochs)...", flush=True)
    model = word2vec.train_word2vec(sentences, 200)
    model.save(str(output_path))
    elapsed = time.monotonic() - started
    print(f"Saved: {output_path}", flush=True)
    print(f"Vocabulary size: {len(model.wv.key_to_index)}", flush=True)
    print(f"Elapsed seconds: {elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
