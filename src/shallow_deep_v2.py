
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# Parallel folds run in separate worker processes, which do not inherit
# warnings.filterwarnings() from this process. The environment variable is
# inherited, so it also silences ConvergenceWarning (a UserWarning) there.
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of src/)
sys.path.insert(0, str(ROOT / "src"))
from mnist_loader import load_images, load_labels  # noqa: E402
from sai_loader import load_sai, make_tfidf  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------- settings --
QUICK = False            # True: tiny test run (2 folds, 3 epochs, 2 configs per experiment)
EXPERIMENTS = ["depth", "width", "reg"]   # remove names to skip experiments
N_FOLDS = 5
MAX_EPOCHS = 20
N_JOBS = -1              # folds in parallel; use 2 if the computer becomes too slow

BASE = dict(alpha=1e-4)  # default MLP settings shared by every configuration

# Each configuration is a dict: "hidden" is required, other keys override
# MLP settings, and "max_features" changes the TF-IDF size (SAI only).
CONFIGS = {
    "depth": {
        "L=1": dict(hidden=(128,)),
        "L=2": dict(hidden=(128,) * 2),
        "L=3": dict(hidden=(128,) * 3),
        "L=5": dict(hidden=(128,) * 5),
        "L=8": dict(hidden=(128,) * 8),
    },
    "width": {
        "32 units": dict(hidden=(32,)),
        "64 units": dict(hidden=(64,)),
        "128 units": dict(hidden=(128,)),   # same model as depth L=1, reused
        "256 units": dict(hidden=(256,)),
    },
    "reg": {
        "baseline (alpha=1e-4)": dict(hidden=(128,)),   # reused from depth L=1
        "alpha=1e-3": dict(hidden=(128,), alpha=1e-3),
        "alpha=1e-2": dict(hidden=(128,), alpha=1e-2),
        "alpha=1e-1": dict(hidden=(128,), alpha=1e-1),
        "early stopping": dict(hidden=(128,), early_stopping=True,
                               n_iter_no_change=3, max_iter=50),
        "TF-IDF 5000 features": dict(hidden=(128,), max_features=5000),
    },
}
DATASETS_PER_EXPERIMENT = {"depth": ["MNIST", "SAI"],
                           "width": ["MNIST", "SAI"],
                           "reg": ["SAI"]}

if QUICK:
    N_FOLDS, MAX_EPOCHS = 2, 3
    CONFIGS = {name: dict(list(cfgs.items())[:2]) for name, cfgs in CONFIGS.items()}


# ------------------------------------------------------------------ models --
def make_mlp(hidden, **overrides):
    """Fully connected network: ReLU hidden layers, softmax output.
    alpha is the L2 penalty added to the loss (weight decay)."""
    params = dict(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=128,          # mini-batch size
        max_iter=MAX_EPOCHS,     # maximum number of epochs
        random_state=0,
        **BASE,
    )
    params.update(overrides)
    return MLPClassifier(**params)


def build(dataset, cfg):
    cfg = dict(cfg)
    hidden = cfg.pop("hidden")
    max_features = cfg.pop("max_features", 20000)
    mlp = make_mlp(hidden, **cfg)
    if dataset == "SAI":
        # TF-IDF inside the pipeline: fitted on the training folds only.
        return make_pipeline(make_tfidf(max_features=max_features), mlp)
    return mlp


def count_params(mlp):
    """Weights + biases of a fitted MLP."""
    return sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)


# ------------------------------------------------------------- experiment --
def evaluate(dataset, X, y, cfg):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    start = time.time()
    scores = cross_validate(
        build(dataset, cfg), X, y, cv=cv,
        scoring={"accuracy": "accuracy", "neg_log_loss": "neg_log_loss"},
        return_train_score=True,
        return_estimator=True,
        n_jobs=N_JOBS,
        error_score="raise",
    )
    mlps = [est[-1] if hasattr(est, "steps") else est for est in scores["estimator"]]
    train_acc = scores["train_accuracy"].mean()
    val_acc = scores["test_accuracy"].mean()
    metrics = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_std": scores["test_accuracy"].std(),
        "gap": train_acc - val_acc,
        "train_log_loss": -scores["train_neg_log_loss"].mean(),
        "val_log_loss": -scores["test_neg_log_loss"].mean(),
        "n_params": count_params(mlps[0]),
        "epochs_run": np.mean([m.n_iter_ for m in mlps]),
        "seconds": round(time.time() - start, 1),
    }
    curves = [(fold, epoch + 1, loss)
              for fold, m in enumerate(mlps) for epoch, loss in enumerate(m.loss_curve_)]
    return metrics, curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP experiments for Question 3.1")
    parser.add_argument("--exp", nargs="+", choices=list(CONFIGS), default=EXPERIMENTS,
                        help="experiments to run (default: all)")
    parser.add_argument("--data", nargs="+", choices=["MNIST", "SAI"], default=["MNIST", "SAI"],
                        help="datasets to run (default: both)")
    parser.add_argument("--quick", action="store_true", help="tiny test run")
    args = parser.parse_args()
    if args.quick and not QUICK:
        QUICK = True
        N_FOLDS, MAX_EPOCHS = 2, 3
        CONFIGS = {name: dict(list(cfgs.items())[:2]) for name, cfgs in CONFIGS.items()}
    EXPERIMENTS = args.exp
    DATASETS_PER_EXPERIMENT = {e: [d for d in ds if d in args.data]
                               for e, ds in DATASETS_PER_EXPERIMENT.items()}

    data = {}
    needed = {d for e in EXPERIMENTS for d in DATASETS_PER_EXPERIMENT[e]}
    if not needed:
        sys.exit(f"Nothing to run: experiment(s) {EXPERIMENTS} do not use dataset(s) {args.data}.")
    if "MNIST" in needed:
        X = load_images(ROOT / "data/mnist/train-images-idx3-ubyte.gz") / 255.0
        y = load_labels(ROOT / "data/mnist/train-labels-idx1-ubyte.gz")
        data["MNIST"] = (X, y)       # only the 60,000 training images
    if "SAI" in needed:
        texts, y, _ = load_sai(ROOT / "data/sai/train.csv")
        data["SAI"] = (np.array(texts, dtype=object), y)

    out_dir = ROOT / "results/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_" + "-".join(EXPERIMENTS) + "_" + "-".join(sorted(needed))
    suffix += "_quick" if QUICK else ""
    results_file = out_dir / f"experiments_cv{suffix}.csv"
    curves_file = out_dir / f"loss_curves{suffix}.csv"
    print("Results will be saved to:", results_file, flush=True)

    rows, curve_rows, cache = [], [], {}
    for experiment in EXPERIMENTS:
        for dataset in DATASETS_PER_EXPERIMENT[experiment]:
            X, y = data[dataset]
            for name, cfg in CONFIGS[experiment].items():
                key = (dataset, tuple(sorted(cfg.items())))
                reused = key in cache
                if not reused:   # identical model already trained -> reuse it
                    cache[key] = evaluate(dataset, X, y, cfg)
                metrics, curves = cache[key]
                rows.append({"experiment": experiment, "dataset": dataset,
                             "config": name, "hidden_layers": str(cfg["hidden"]),
                             **metrics})
                if not reused:
                    curve_rows += [{"experiment": experiment, "dataset": dataset,
                                    "config": name, "fold": f, "epoch": e, "loss": l}
                                   for f, e, l in curves]
                m = metrics
                print(f"{experiment:5s} | {dataset:5s} | {name:22s} | "
                      f"train {m['train_acc']:.4f} | val {m['val_acc']:.4f} ± {m['val_std']:.4f} | "
                      f"gap {m['gap']:.4f} | val logloss {m['val_log_loss']:.3f} | "
                      f"params {m['n_params']:,} | epochs {m['epochs_run']:.1f} | "
                      f"{'reused' if reused else str(m['seconds']) + ' s'}", flush=True)
                # Save after every configuration, so finished results are kept
                # even if the run stops halfway.
                pd.DataFrame(rows).to_csv(results_file, index=False)
                pd.DataFrame(curve_rows).to_csv(curves_file, index=False)

    print("Saved:", results_file)
    print("Saved:", curves_file)