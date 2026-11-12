"""Adapt DRWeiboV3 JSON propagation trees to BiGCN's official graph format.

Why this script exists
----------------------
BiGCN (AAAI 2020) ships Weibo data as `weibotree.txt`: one row per node,
`eid \\t parent_index \\t node_index \\t "vocab_index:count ..."`, with a fixed
5000-dim bag-of-words node feature (`Net(5000, 64, 64)` in the official model).
DRWeiboV3 is a different dataset shipped as JSON. To compare on the SAME
dataset and the SAME 802 split as RAGCL / AD-GSCL, we convert it here.

What is preserved
-----------------
* The exact 802 split: filenames are read from RAGCL's official
  `Data/DRWeiboV3/dataset/{train,val,test}/raw/*.json`, produced by official
  `Main/sort.py` with `random.seed(1234)`. No re-shuffling, no re-splitting.
* Graph topology: root = node 0; comment `comment id` k -> node k+1;
  `parent = -1` -> root, `parent = p` -> node p+1. Edges are top-down
  (parent -> child), identical to RAGCL's `Main/dataset.py` row/col mapping.
  BiGCN's `BiGraphDataset` derives BU edges by reversing these itself.
* Labels: DRWeiboV3 0/1 binary, same as BiGCN's 2-class Weibo head.

What is an ADAPTATION (must be reported)
---------------------------------------
* 5000-dim bag-of-words vocabulary is BUILT HERE with jieba on TRAIN texts
  only (top 5000 by document frequency), because the official preprocessed
  Weibo vocabulary is unavailable. Values are raw term counts, like the
  official `index:count` format. This is NOT the authors' vocabulary.
* npz files use `np.savez_compressed` + float32 instead of the official dense
  `np.savez`, purely to keep the ~7.6 GB dense feature matrix manageable.
  Contents are identical after `np.load`.
* BiGCN's official 5-fold CV and EarlyStopping are NOT used; see
  run_bigcn_drweibo.py for the unified 802 / 10x100 protocol.

No BiGCN model source is modified by this script.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import jieba
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "RAGCL" / "Data" / "DRWeiboV3" / "source"
SPLIT_ROOT = ROOT / "RAGCL" / "Data" / "DRWeiboV3" / "dataset"
OUT = ROOT / "BiGCN" / "data" / "DRWeiboV3graph"
VOCAB_SIZE = 5000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def split_ids(split):
    return sorted(p.stem for p in (SPLIT_ROOT / split / "raw").glob("*.json"))


def node_texts(post):
    """Root first, then comments ordered by their contiguous `comment id`."""
    comments = sorted(post["comment"], key=lambda c: c["comment id"])
    return [str(post["source"]["content"])] + [str(c["content"]) for c in comments]


def bow(text, vocab):
    counts = Counter()
    for tok in jieba.lcut(text):
        idx = vocab.get(tok)
        if idx is not None:
            counts[idx] += 1
    return counts


def build_vocab(ids):
    df = Counter()
    for eid in ids:
        post = read_json(SOURCE / f"{eid}.json")
        for text in node_texts(post):
            df.update(set(jieba.lcut(text)))
    # deterministic: sort by (-document frequency, token) before truncation
    ranked = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:VOCAB_SIZE]
    return {tok: i for i, (tok, _) in enumerate(ranked)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {s: split_ids(s) for s in ("train", "val", "test")}
    print("split sizes:", {k: len(v) for k, v in splits.items()}, flush=True)

    vocab = build_vocab(splits["train"])
    print("vocab size:", len(vocab), flush=True)
    (out / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    all_ids = sorted({eid for ids in splits.values() for eid in ids})
    tree_dic = {}
    labels = {}
    for i, eid in enumerate(all_ids):
        post = read_json(SOURCE / f"{eid}.json")
        texts = node_texts(post)
        n = len(texts)
        x = np.zeros((n, VOCAB_SIZE), dtype=np.float32)
        for row, text in enumerate(texts):
            for idx, cnt in bow(text, vocab).items():
                x[row, idx] = cnt
        root = x[0:1].copy()

        row, col = [], []
        for c in post["comment"]:
            row.append(c["parent"] + 1)
            col.append(c["comment id"] + 1)
        edgeindex = np.array([row, col], dtype=np.int64)

        np.savez_compressed(
            out / f"{eid}.npz",
            x=x,
            root=root,
            edgeindex=edgeindex,
            rootindex=np.array(0, dtype=np.int64),
            y=np.array(post["source"]["label"], dtype=np.int64),
        )
        tree_dic[eid] = n  # BiGraphDataset only uses len(); scalar is enough
        labels[eid] = int(post["source"]["label"])
        if (i + 1) % 1000 == 0:
            print(f"built {i + 1}/{len(all_ids)} graphs", flush=True)

    (out / "split.json").write_text(
        json.dumps({"splits": splits, "tree_dic": tree_dic, "labels": labels},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    total_mb = sum(p.stat().st_size for p in out.glob("*.npz")) / 1048576
    print(f"done: {len(all_ids)} graphs, {total_mb:.1f} MB npz", flush=True)


if __name__ == "__main__":
    main()
