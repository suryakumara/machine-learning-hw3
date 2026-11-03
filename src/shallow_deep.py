"""Experiment (a)(b): shallow vs deep fully connected networks on MNIST and SAI.

Runs stratified 5-fold cross-validation for several network depths and saves
the mean training and validation accuracy to results/tables/shallow_deep_cv.csv.
Run from anywhere:  python src/shallow_deep.py
"""
import sys
import time
import warnings
from pathlib import Path

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

# Set QUICK = True for a fast test run (fewer configurations, 2 folds, few epochs).
QUICK = False

CONFIGS = {
    "1 hidden layer": (128,),
    "2 hidden layers": (128, 128),
    "3 hidden layers": (128, 128, 128),
    "5 hidden layers": (128,) * 5,
    "8 hidden layers": (128,) * 8,
}
N_FOLDS = 5
MAX_EPOCHS = 20

if QUICK:
    CONFIGS = {"1 hidden layer": (64,), "3 hidden layers": (64, 64, 64)}
    N_FOLDS = 2
    MAX_EPOCHS = 3


def make_mlp(hidden):
    """Fully connected network. alpha is the L2 penalty (weight decay)."""
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=128,          # mini-batch size
        max_iter=MAX_EPOCHS,     # number of epochs
        alpha=1e-4,              # weight decay strength
        random_state=0,
    )


def run(dataset_name, X, y, build_model):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    rows = []
    for config_name, hidden in CONFIGS.items():
        start = time.time()
        scores = cross_validate(build_model(hidden), X, y, cv=cv,
                                return_train_score=True, n_jobs=-1)
        row = {
            "dataset": dataset_name,
            "config": config_name,
            "hidden_layers": str(hidden),
            "train_acc": scores["train_score"].mean(),
            "val_acc": scores["test_score"].mean(),
            "val_std": scores["test_score"].std(),
            "seconds": round(time.time() - start, 1),
        }
        rows.append(row)
        print(f"{dataset_name:5s} | {config_name:16s} | train {row['train_acc']:.4f} "
              f"| val {row['val_acc']:.4f} ± {row['val_std']:.4f} | {row['seconds']} s")
    return rows


if __name__ == "__main__":
    results = []

    # MNIST: only the 60,000 training images are used for cross-validation.
    X_mnist = load_images(ROOT / "data/mnist/train-images-idx3-ubyte.gz") / 255.0
    y_mnist = load_labels(ROOT / "data/mnist/train-labels-idx1-ubyte.gz")
    results += run("MNIST", X_mnist, y_mnist, make_mlp)

    # SAI: TF-IDF is inside the pipeline, so it is fitted on the training folds only.
    texts, y_sai, _ = load_sai(ROOT / "data/sai/train.csv")
    results += run("SAI", np.array(texts, dtype=object), y_sai,
                   lambda hidden: make_pipeline(make_tfidf(), make_mlp(hidden)))

    out_dir = ROOT / "results/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / ("shallow_deep_cv_quick.csv" if QUICK else "shallow_deep_cv.csv")
    pd.DataFrame(results).to_csv(out_file, index=False)
    print("Saved:", out_file)