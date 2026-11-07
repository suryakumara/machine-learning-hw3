"""Question 3.1 (h): can foundation models give similar or better results?

Three methods, compared with the MLP baselines from shallow_deep_v2.py and
final_test.py (same 5-fold StratifiedKFold splits, random_state=0):

  minilm      SAI   frozen all-MiniLM-L6-v2 sentence embeddings + logistic regression
  distilbert  SAI   fine-tuned distilbert-base-cased (all weights are trained)
  dinov2      MNIST frozen DINOv2 ViT-S/14 features + logistic regression
                    (5-fold CV on the 60,000 training images, then one fit on all
                    of them and evaluation on the 10,000 test images)

Frozen features are computed once and cached in models/features/ (ignored by git),
so the logistic-regression part can be re-run quickly.

Examples (run from anywhere):
  python src/foundation_model_h.py --quick                      # small test of all methods
  python src/foundation_model_h.py --method minilm dinov2       # fast ones first
  python src/foundation_model_h.py --method distilbert          # slow: ~20-30 min per fold on M1
  python src/foundation_model_h.py --method distilbert --folds 0 1   # only some folds

Finished folds are kept in results/tables/foundation_model_folds.csv and skipped
on the next run, so a stopped run (e.g. a Colab disconnect) can be continued.
Use --redo to recompute them.
"""
import argparse
import contextlib
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mnist_loader import load_images, load_labels  # noqa: E402
from sai_loader import load_sai  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "distilbert": "distilbert-base-cased",   # cased: capitals and punctuation are style cues
    "dinov2": "facebook/dinov2-small",       # ViT-S/14
}
DATASET = {"minilm": "SAI", "distilbert": "SAI", "dinov2": "MNIST"}

parser = argparse.ArgumentParser(description="Foundation models for Question 3.1 (h)")
parser.add_argument("--method", nargs="+", choices=list(MODELS), default=list(MODELS))
parser.add_argument("--folds", nargs="+", type=int, default=None,
                    help="CV folds to run (default: all). The MNIST test evaluation runs "
                         "when no --folds are given.")
parser.add_argument("--n-folds", type=int, default=5)
parser.add_argument("--quick", action="store_true",
                    help="tiny test run: small data subset, 2 folds, 1 epoch")
parser.add_argument("--redo", action="store_true", help="recompute folds already saved")
# DistilBERT fine-tuning (fixed beforehand, not tuned on the validation folds)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--max-len", type=int, default=128, help="max tokens per sentence")
# Frozen features
parser.add_argument("--img-size", type=int, default=224,
                    help="MNIST images are resized to this (multiple of 14); 112 is ~4x faster")
parser.add_argument("--embed-batch", type=int, default=128)
parser.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the LR")
parser.add_argument("--n-jobs", type=int, default=-1, help="parallel folds for the LR")
args = parser.parse_args()
if args.quick:
    args.n_folds, args.epochs = 2, 1


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
print("Device:", DEVICE, flush=True)

OUT_DIR = ROOT / "results/tables"
CACHE_DIR = ROOT / "models/features"
SUFFIX = "_quick" if args.quick else ""
FOLDS_FILE = OUT_DIR / f"foundation_model_folds{SUFFIX}.csv"
SUMMARY_FILE = OUT_DIR / f"foundation_model{SUFFIX}.csv"
CURVES_FILE = OUT_DIR / f"foundation_model_distilbert_curves{SUFFIX}.csv"


def autocast():
    """Mixed precision on CUDA (e.g. Colab T4); full precision on MPS / CPU."""
    if DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_cv():
    return StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=0)


def probs_metrics(y, proba):
    proba = np.asarray(proba, dtype=np.float64)
    proba /= proba.sum(axis=1, keepdims=True)
    return accuracy_score(y, proba.argmax(1)), log_loss(y, proba, labels=np.arange(proba.shape[1]))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# -------------------------------------------------------------------- data --
def load_sai_data():
    texts, y, _ = load_sai(ROOT / "data/sai/train.csv")
    texts, y = list(texts), np.asarray(y, dtype=np.int64)
    if args.quick:
        keep = np.random.RandomState(0).choice(len(y), 2000, replace=False)
        texts, y = [texts[i] for i in keep], y[keep]
    return texts, y


def load_mnist_data():
    Xtr = load_images(ROOT / "data/mnist/train-images-idx3-ubyte.gz")
    ytr = load_labels(ROOT / "data/mnist/train-labels-idx1-ubyte.gz").astype(np.int64)
    Xte = load_images(ROOT / "data/mnist/t10k-images-idx3-ubyte.gz")
    yte = load_labels(ROOT / "data/mnist/t10k-labels-idx1-ubyte.gz").astype(np.int64)
    if args.quick:
        Xtr, ytr, Xte, yte = Xtr[:3000], ytr[:3000], Xte[:1000], yte[:1000]
    return Xtr, ytr, Xte, yte


# ------------------------------------------------------- frozen features --
@torch.no_grad()
def embed_minilm(texts):
    """Mean-pooled, L2-normalised MiniLM sentence embeddings (384 dims)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["minilm"])
    model = AutoModel.from_pretrained(MODELS["minilm"]).to(DEVICE).eval()
    out = []
    for s in range(0, len(texts), args.embed_batch):
        b = tok(texts[s:s + args.embed_batch], padding=True, truncation=True,
                max_length=256, return_tensors="pt").to(DEVICE)
        with autocast():
            h = model(**b).last_hidden_state.float()
        mask = b["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # mean over real tokens
        out.append(F.normalize(emb, dim=1).cpu().numpy())
        print(f"\r  MiniLM embeddings {min(s + args.embed_batch, len(texts))}/{len(texts)}",
              end="", flush=True)
    print()
    return np.concatenate(out), count_params(model)


@torch.no_grad()
def embed_dinov2(images):
    """DINOv2 features: [CLS token, mean of patch tokens] -> 768 dims.
    28x28 grey digits are resized, copied to 3 channels and ImageNet-normalised."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODELS["dinov2"]).to(DEVICE).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    out = []
    for s in range(0, len(images), args.embed_batch):
        x = torch.from_numpy(images[s:s + args.embed_batch].astype(np.float32) / 255.0)
        x = x.view(-1, 1, 28, 28).to(DEVICE)
        x = F.interpolate(x, size=(args.img_size, args.img_size), mode="bilinear",
                          align_corners=False)
        x = (x.repeat(1, 3, 1, 1) - mean) / std
        with autocast():
            h = model(pixel_values=x).last_hidden_state.float()
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).cpu().numpy())
        print(f"\r  DINOv2 features {min(s + args.embed_batch, len(images))}/{len(images)}",
              end="", flush=True)
    print()
    return np.concatenate(out), count_params(model)


def cached(name, fn, *inputs):
    """Compute features once; later runs load them from models/features/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}{SUFFIX}.npz"
    if path.exists():
        d = np.load(path)
        print(f"  loaded cached features {path.name} {d['X'].shape}", flush=True)
        return d["X"], int(d["n_params"])
    start = time.time()
    X, n_params = fn(*inputs)
    np.savez(path, X=X, n_params=n_params)
    print(f"  features {X.shape} in {time.time() - start:.1f} s -> {path}", flush=True)
    return X, n_params


def make_probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=2000))


def probe_cv(method, X, y, backbone_params):
    """Logistic regression on frozen features, same CV splits as the MLP baselines.
    The features never saw the labels, so computing them on all data is no leakage."""
    start = time.time()
    s = cross_validate(make_probe(), X, y, cv=make_cv(), n_jobs=args.n_jobs,
                       scoring={"accuracy": "accuracy", "neg_log_loss": "neg_log_loss"},
                       return_train_score=True)
    seconds = (time.time() - start) / args.n_folds
    n_classes = len(np.unique(y))
    return [{"method": method, "dataset": DATASET[method], "split": "cv", "fold": f,
             "train_acc": s["train_accuracy"][f], "val_acc": s["test_accuracy"][f],
             "train_log_loss": -s["train_neg_log_loss"][f],
             "val_log_loss": -s["test_neg_log_loss"][f],
             "trainable_params": X.shape[1] * n_classes + n_classes,
             "total_params": backbone_params + X.shape[1] * n_classes + n_classes,
             "seconds": round(seconds, 1)}
            for f in range(args.n_folds)]


def run_minilm():
    texts, y = load_sai_data()
    X, n_params = cached("minilm_SAI", embed_minilm, texts)
    return probe_cv("minilm", X, y, n_params)


def run_dinov2():
    Xtr, ytr, Xte, yte = load_mnist_data()
    images = np.concatenate([Xtr, Xte])
    F_all, n_params = cached(f"dinov2_MNIST_{args.img_size}px", embed_dinov2, images)
    Ftr, Fte = F_all[:len(Xtr)], F_all[len(Xtr):]
    rows = probe_cv("dinov2", Ftr, ytr, n_params)
    if args.folds is None:     # final evaluation, like final_test.py
        start = time.time()
        probe = make_probe().fit(Ftr, ytr)
        tr_acc, tr_ll = probs_metrics(ytr, probe.predict_proba(Ftr))
        te_acc, te_ll = probs_metrics(yte, probe.predict_proba(Fte))
        rows.append({"method": "dinov2", "dataset": "MNIST", "split": "test", "fold": -1,
                     "train_acc": tr_acc, "val_acc": te_acc,
                     "train_log_loss": tr_ll, "val_log_loss": te_ll,
                     "trainable_params": rows[0]["trainable_params"],
                     "total_params": rows[0]["total_params"],
                     "seconds": round(time.time() - start, 1)})
        print(f"  dinov2 | MNIST test | acc {te_acc:.4f} | log loss {te_ll:.4f}", flush=True)
    return rows


# ------------------------------------------------- DistilBERT fine-tuning --
def run_distilbert(done):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    texts, y = load_sai_data()
    tok = AutoTokenizer.from_pretrained(MODELS["distilbert"])
    enc = tok(texts, truncation=True, max_length=args.max_len)   # no padding yet
    n_cut = sum(len(ids) == args.max_len for ids in enc["input_ids"])
    print(f"  {n_cut} of {len(texts)} sentences truncated to {args.max_len} tokens", flush=True)

    def collate(idx):
        b = tok.pad({"input_ids": [enc["input_ids"][i] for i in idx],
                     "attention_mask": [enc["attention_mask"][i] for i in idx]},
                    return_tensors="pt")
        return {k: v.to(DEVICE) for k, v in b.items()}

    @torch.no_grad()
    def predict_proba(model, idx):
        model.eval()
        idx = np.asarray(idx)
        order = np.argsort([len(enc["input_ids"][i]) for i in idx])   # less padding
        proba = np.zeros((len(idx), 3))
        for s in range(0, len(idx), args.batch * 2):
            pos = order[s:s + args.batch * 2]
            with autocast():
                logits = model(**collate(idx[pos])).logits
            proba[pos] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return proba

    rows, curves = [], []
    for fold, (tr, va) in enumerate(make_cv().split(np.zeros(len(y)), y)):
        if args.folds is not None and fold not in args.folds:
            continue
        if ("distilbert", fold) in done:
            print(f"  distilbert fold {fold} already saved, skipping (use --redo)", flush=True)
            continue
        start = time.time()
        torch.manual_seed(fold)
        gen = torch.Generator().manual_seed(fold)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODELS["distilbert"], num_labels=3).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = args.epochs * int(np.ceil(len(tr) / args.batch))
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
        y_t = torch.from_numpy(y).to(DEVICE)

        for epoch in range(1, args.epochs + 1):
            model.train()
            perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
            loss_sum, n = 0.0, 0
            for s in range(0, len(perm), args.batch):
                idx = perm[s:s + args.batch]
                with autocast():
                    loss = F.cross_entropy(model(**collate(idx)).logits.float(), y_t[idx])
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                loss_sum += loss.item()
                n += 1
                if n % 50 == 0:
                    print(f"\r  fold {fold} epoch {epoch} step {n}/{int(np.ceil(len(tr) / args.batch))}"
                          f" loss {loss_sum / n:.4f} ({time.time() - start:.0f} s)",
                          end="", flush=True)
            print()
            va_acc, va_ll = probs_metrics(y[va], predict_proba(model, va))
            curves.append({"fold": fold, "epoch": epoch, "train_loss": loss_sum / n,
                           "val_acc": va_acc, "val_log_loss": va_ll})
            print(f"  distilbert fold {fold} epoch {epoch} | train loss {loss_sum / n:.4f} | "
                  f"val acc {va_acc:.4f} | val log loss {va_ll:.4f}", flush=True)

        tr_acc, tr_ll = probs_metrics(y[tr], predict_proba(model, tr))
        n_params = count_params(model)
        rows.append({"method": "distilbert", "dataset": "SAI", "split": "cv", "fold": fold,
                     "train_acc": tr_acc, "val_acc": va_acc,
                     "train_log_loss": tr_ll, "val_log_loss": va_ll,
                     "trainable_params": n_params, "total_params": n_params,
                     "seconds": round(time.time() - start, 1)})
        # Save after every fold (one fold can take ~half an hour on M1).
        save(rows, "distilbert")
        save_curves(curves)
        rows = []
        del model, opt
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "mps":
            torch.mps.empty_cache()
    return rows


# ----------------------------------------------------------------- saving --
def read_folds():
    return pd.read_csv(FOLDS_FILE) if FOLDS_FILE.exists() else pd.DataFrame()


def save(rows, method):
    """Merge new fold rows into the folds file and rebuild the summary."""
    if not rows:
        return
    new = pd.DataFrame(rows)
    old = read_folds()
    if len(old):
        key = ["method", "split", "fold"]
        old = old.merge(new[key], on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
    folds = pd.concat([old, new], ignore_index=True).sort_values(["method", "split", "fold"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_FILE, index=False)
    summarize(folds)


def save_curves(curves):
    new = pd.DataFrame(curves)
    if CURVES_FILE.exists():
        old = pd.read_csv(CURVES_FILE)
        old = old[~old["fold"].isin(new["fold"])]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["fold", "epoch"]).to_csv(CURVES_FILE, index=False)


def summarize(folds):
    """One row per method, with the same columns as experiments_cv_*.csv where possible."""
    rows = []
    for method, g in folds.groupby("method", sort=False):
        cv, test = g[g["split"] == "cv"], g[g["split"] == "test"]
        row = {"method": method, "dataset": DATASET[method], "model": MODELS[method],
               "folds_done": len(cv),
               "train_acc": cv["train_acc"].mean(), "val_acc": cv["val_acc"].mean(),
               "val_std": cv["val_acc"].std(ddof=0),
               "gap": cv["train_acc"].mean() - cv["val_acc"].mean(),
               "train_log_loss": cv["train_log_loss"].mean(),
               "val_log_loss": cv["val_log_loss"].mean(),
               "trainable_params": int(cv["trainable_params"].iloc[0]),
               "total_params": int(cv["total_params"].iloc[0]),
               "seconds_per_fold": cv["seconds"].mean()}
        if len(test):
            row["test_acc"] = test["val_acc"].iloc[0]
            row["test_log_loss"] = test["val_log_loss"].iloc[0]
        rows.append(row)
    pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False)


def print_summary():
    if not SUMMARY_FILE.exists():
        return
    print("\nSummary (" + str(SUMMARY_FILE) + "):")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.read_csv(SUMMARY_FILE).drop(columns="model").round(4).to_string(index=False))


# ------------------------------------------------------------------- main --
if __name__ == "__main__":
    existing = read_folds()
    done = set() if args.redo or existing.empty else \
        set(zip(existing.loc[existing["split"] == "cv", "method"],
                existing.loc[existing["split"] == "cv", "fold"]))

    for method in args.method:
        print(f"\n=== {method} ({MODELS[method]}) on {DATASET[method]} ===", flush=True)
        if method == "distilbert":
            run_distilbert(done)
            continue
        all_folds = set(range(args.n_folds))
        wanted = all_folds if args.folds is None else all_folds & set(args.folds)
        if not args.redo and wanted <= {f for m, f in done if m == method}:
            print(f"  {method} already saved, skipping (use --redo)", flush=True)
            continue
        rows = run_minilm() if method == "minilm" else run_dinov2()
        # The LR runs on all folds at once (fast); keep only the requested ones.
        rows = [r for r in rows if r["split"] == "test" or r["fold"] in wanted]
        for r in rows:
            if r["split"] == "cv":
                print(f"  {method} fold {r['fold']} | train {r['train_acc']:.4f} | "
                      f"val {r['val_acc']:.4f} | val log loss {r['val_log_loss']:.4f}", flush=True)
        save(rows, method)

    print_summary()
    print("\nSaved:", FOLDS_FILE)
