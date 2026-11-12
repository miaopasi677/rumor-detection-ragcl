"""Run official BiGCN on DRWeiboV3 under the SAME 802 / 10x100 protocol as RAGCL / AD-GSCL.

Model code is NOT modified: the three official classes TDrumorGCN, BUrumorGCN
and Net are loaded verbatim from BiGCN/model/Weibo/BiGCN_Weibo.py (lines
16-89, before the module-level training loop that would otherwise execute on
import). Official hyperparameters are kept: lr 0.0005, weight_decay 1e-4,
batchsize 16, hidden 64, Adam with BU conv lr/5, nll_loss, Net(5000, 64, 64).

Protocol ADAPTATIONS (must be reported):
  * Official 5-fold CV  -> the exact 802 split filenames from RAGCL's
    Main/sort.py output (train 4829 / val 1 / test 1208).
  * Official 200 epochs + EarlyStopping -> fixed 100 epochs, no early stop;
    headline = mean test ACC over the last 10 epochs (RAGCL/AD-GSCL convention).
  * 10 independent runs (official reports `iterations` x 5 folds).
  * num_workers 10 -> 0 (Windows DataLoader), cuda:1 -> cuda:0 (single GPU).
  * DataLoader shuffle follows the unified protocol (train=True, test=False);
    official BiGCN uses train=False, test=True. This only changes sample order,
    not the model, loss, split or metric.
  * Node features: 5000-dim bag-of-words built by build_bigcn_drweibo.py
    (jieba, TRAIN-only vocab), instead of the authors' preprocessed Weibo
    vocabulary. This is a real feature difference and limits comparability
    with the published BiGCN number.
  * `root` is kept as float32 features (official casts it to LongTensor); the
    official forward() never reads data.root, it uses data.rootindex instead.
  * RUNTIME ONLY, numerically identical: the official classes are exec'd with a
    thin `th` proxy whose `zeros()` allocates directly on `device`. The official
    line `th.zeros(n, 5000).to(device)` otherwise allocates 20-32 MB on the CPU
    four times per batch; with 3-4 parallel processes on a 16 GB machine that
    exhausts physical RAM. `torch.zeros(n, m, device=cuda)` and
    `torch.zeros(n, m).to(cuda)` produce the SAME tensor, so this changes only
    where the temporary lives, not any value, gradient, parameter or loss.

Memory note: all 6037 dense x tensors would need ~7.6 GB RAM. We cache only
scipy CSR matrices (~40 MB total) and materialise a single dense graph in
__getitem__, so every run stays bounded regardless of dataset size.

No official source file is edited. All outputs are real logs.
"""

import argparse
import ast
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.data import Batch, Data, DataLoader

ROOT = Path(__file__).resolve().parent
BIGCN = ROOT / "BiGCN"
GRAPH_DIR = BIGCN / "data" / "DRWeiboV3graph"
MODEL_FILE = BIGCN / "model" / "Weibo" / "BiGCN_Weibo.py"
LOG_DIR = BIGCN / "Log"
CONSOLE_DIR = BIGCN / "logs"

VOCAB_SIZE = 5000
LR = 0.0005          # official
WEIGHT_DECAY = 1e-4  # official
BATCHSIZE = 16       # official
HIDDEN = 64          # official


class _ThProxy:
    """Forward every torch attribute, except zeros() which allocates on GPU.

    The official forward() calls `th.zeros(n, m).to(device)`. On a CPU-bound
    host with several parallel processes this temporary is what exhausts RAM.
    Allocating it on `device` directly is numerically identical.
    """

    def __init__(self, device):
        self._device = device

    def __getattr__(self, name):
        return getattr(torch, name)

    def zeros(self, *size, **kwargs):
        kwargs.setdefault("device", self._device)
        return torch.zeros(*size, **kwargs)


def load_official_net(device):
    """Exec the official class definitions verbatim, without the training loop."""
    source = MODEL_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"TDrumorGCN", "BUrumorGCN", "Net"}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
    if len(nodes) != 3:
        raise RuntimeError(f"Expected 3 official model classes, found {len(nodes)}")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "th": _ThProxy(device),
        "torch": torch,
        "F": F,
        "np": np,
        "copy": __import__("copy"),
        "nn": torch.nn,
        "device": device,
        "scatter_mean": __import__("torch_scatter").scatter_mean,
        "GCNConv": __import__("torch_geometric.nn", fromlist=["GCNConv"]).GCNConv,
    }
    exec(compile(module, str(MODEL_FILE), "exec"), ns)
    return ns["Net"]


class GraphList:
    """Lazy loader with a SPARSE cache; dense tensors exist one graph at a time."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.cache = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        eid = self.ids[i]
        if eid not in self.cache:
            with np.load(GRAPH_DIR / f"{eid}.npz") as d:
                self.cache[eid] = {
                    "x": csr_matrix(d["x"]),
                    "root": csr_matrix(d["root"]),
                    "edgeindex": np.asarray(d["edgeindex"], dtype=np.int64).copy(),
                    "y": int(d["y"]),
                    "rootindex": int(d["rootindex"]),
                }
        d = self.cache[eid]
        x = torch.from_numpy(d["x"].toarray().astype(np.float32))
        root = torch.from_numpy(d["root"].toarray().astype(np.float32))
        edge = torch.from_numpy(d["edgeindex"])
        return Data(
            x=x,
            edge_index=edge,
            BU_edge_index=torch.stack([edge[1], edge[0]]),
            y=torch.LongTensor([d["y"]]),
            root=root,
            rootindex=torch.LongTensor([d["rootindex"]]),
        )


def collate(batch):
    return Batch.from_data_list(batch)


def check_batch_increments(ds):
    """Verify PyG adds node offsets to rootindex / BU_edge_index on collate."""
    b = Batch.from_data_list([ds[0], ds[1]])
    n0 = int(ds[0].x.size(0))
    if int(b.rootindex[1]) != n0:
        raise RuntimeError(
            f"Batch collate did NOT offset rootindex: got {b.rootindex.tolist()}, "
            f"expected [0, {n0}]. Official Net would read the wrong root feature."
        )
    if int(b.BU_edge_index[0][0]) < n0 or int(b.BU_edge_index[1][0]) < n0:
        # first edges belong to graph 0, so only graph 1 edges are offset
        if int(b.BU_edge_index.max()) < n0:
            raise RuntimeError("BU_edge_index was not offset by num_nodes")
    return n0


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            out = model(b)
            ys.extend(b.y.view(-1).tolist())
            ps.extend(out.max(1).indices.tolist())
    ys, ps = np.array(ys), np.array(ps)
    return {
        "acc": round(float(accuracy_score(ys, ps)), 4),
        "macro_p": round(float(precision_score(ys, ps, average="macro", zero_division=0)), 4),
        "macro_r": round(float(recall_score(ys, ps, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(ys, ps, average="macro", zero_division=0)), 4),
        "per_class_p": [round(float(v), 4) for v in precision_score(
            ys, ps, labels=[0, 1], average=None, zero_division=0)],
        "per_class_r": [round(float(v), 4) for v in recall_score(
            ys, ps, labels=[0, 1], average=None, zero_division=0)],
        "per_class_f1": [round(float(v), 4) for v in f1_score(
            ys, ps, labels=[0, 1], average=None, zero_division=0)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="drweibo", help="log filename tag")
    ap.add_argument("--run-offset", type=int, default=0, help="run index offset for resume")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    meta = json.loads((GRAPH_DIR / "split.json").read_text(encoding="utf-8"))
    train_ids, test_ids = meta["splits"]["train"], meta["splits"]["test"]

    Net = load_official_net(device)
    train_ds, test_ds = GraphList(train_ids), GraphList(test_ids)
    n0 = check_batch_increments(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True,
                              num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=False,
                             num_workers=0, collate_fn=collate)
    print(f"device={device} train={len(train_ds)} test={len(test_ds)} "
          f"batch_offset_check={n0} torch_seed={torch.initial_seed()}", flush=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    log_json = LOG_DIR / f"bigcn_{args.tag}_{stamp}.json"
    record = []

    for i in range(args.runs):
        run = args.run_offset + i
        model = Net(VOCAB_SIZE, HIDDEN, HIDDEN).to(device)
        bu = list(map(id, model.BUrumorGCN.conv1.parameters()))
        bu += list(map(id, model.BUrumorGCN.conv2.parameters()))
        base = filter(lambda p: id(p) not in bu, model.parameters())
        optimizer = torch.optim.Adam([
            {"params": base},
            {"params": model.BUrumorGCN.conv1.parameters(), "lr": LR / 5},
            {"params": model.BUrumorGCN.conv2.parameters(), "lr": LR / 5},
        ], lr=LR, weight_decay=WEIGHT_DECAY)

        accs, f1s, ps, rs, losses = [], [], [], [], []
        t0 = time.time()
        for epoch in range(args.epochs + 1):
            if epoch > 0:
                model.train()
                total = 0.0
                for b in train_loader:
                    b = b.to(device)
                    optimizer.zero_grad()
                    out = model(b)
                    loss = F.nll_loss(out, b.y.view(-1))
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * b.num_graphs
                losses.append(round(total / len(train_ds), 6))
            m = evaluate(model, test_loader, device)
            accs.append(m["acc"])
            f1s.append(m["macro_f1"])
            ps.append(m["macro_p"])
            rs.append(m["macro_r"])
            if epoch > 0 and epoch % 10 == 0:
                gc.collect()  # bound Python-side churn over 100 epochs
            if epoch % 10 == 0 or epoch == args.epochs:
                print(f"run {run} epoch {epoch:03d} test_acc={m['acc']} "
                      f"macro_f1={m['macro_f1']} elapsed={time.time() - t0:.0f}s", flush=True)

        tail = accs[-10:]
        k = len(tail)  # 10 for the real 100-epoch protocol; smoke tests use fewer
        row = {
            "run": run,
            "epochs_recorded": len(accs),
            "test accs": accs,
            "macro f1s": f1s,
            "macro ps": ps,
            "macro rs": rs,
            "train losses": losses,
            "final10_mean_acc": round(sum(tail) / k, 4),
            "final10_mean_macro_f1": round(sum(f1s[-10:]) / k, 4),
            "best_epoch_acc": round(max(accs), 4),
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        record.append(row)
        log_json.write_text(json.dumps({
            "model": "BiGCN (official Net, DRWeiboV3 adapted)",
            "protocol": "802 split / 10 runs x 100 epochs / last-10 mean",
            "config": {"lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCHSIZE,
                       "hidden": HIDDEN, "vocab_size": VOCAB_SIZE,
                       "features": "jieba BoW, TRAIN-only top-5000 vocab",
                       "torch_initial_seed": torch.initial_seed(),
                       "device": str(device)},
            "source_logs": "RAGCL Main/sort.py 802 split filenames",
            "runs": args.runs, "epochs": args.epochs, "record": record,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"RUN {run} DONE final10={row['final10_mean_acc']} "
              f"elapsed={row['elapsed_seconds']}s log={log_json.name}", flush=True)

    finals = [r["final10_mean_acc"] for r in record]
    avg = sum(finals) / len(finals)
    sd = math.sqrt(sum((x - avg) ** 2 for x in finals) / len(finals))
    print(f"FINAL n={len(finals)} mean={avg:.4f} population_sd={sd:.4f}", flush=True)


if __name__ == "__main__":
    main()
