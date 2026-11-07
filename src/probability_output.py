"""Experiment (d): does it help to interpret the output units as probabilities?

Compares three models with the same stratified 5-fold cross-validation:

  linear output + squared error   MLP, 1 hidden layer, outputs are plain numbers
                                   trained towards one-hot targets (like the
                                   gradient descent rule with no threshold)
  softmax + cross-entropy          the same MLP, but outputs are probabilities and
                                   the loss is the log loss (logistic-regression kind)
  logistic regression              no hidden layer, softmax output (baseline)

Metrics per model: accuracy, log loss, Brier score (mean squared error between
the predicted probabilities and the one-hot label), and the share of raw outputs
that fall outside [0, 1] - real probabilities never do.

For the squared-error model, which has no probabilities, the outputs are clipped
to a small positive value and rescaled to sum to 1 before log loss and Brier
score are computed. Its accuracy uses the largest output (argmax) and needs no
such fix.

Run from anywhere:
  python src/probability_output.py                 # both datasets
  python src/probability_output.py --data MNIST
Output:
  results/tables/probability_output_<data>.csv
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"   # also silences parallel workers

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mnist_loader import load_images, load_labels  # noqa: E402
from sai_loader import load_sai, make_tfidf  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_FOLDS = 5
MAX_EPOCHS = 20
HIDDEN = (128,)
EPS = 1e-6

# Same MLP settings as shallow_deep.py, so the only difference is the output layer.
MLP_SETTINGS = dict(hidden_layer_sizes=HIDDEN, activation="relu", solver="adam",
                    learning_rate_init=1e-3, batch_size=128, max_iter=MAX_EPOCHS,
                    alpha=1e-4, random_state=0)


class SquaredErrorMLP(ClassifierMixin, BaseEstimator):
    """MLP with a linear output layer trained with squared error on one-hot targets.

    MLPRegressor has an identity output and minimizes 1/2 * sum (y - h(x))^2,
    so its outputs are not forced to be probabilities.
    """

    def __init__(self, hidden_layer_sizes=HIDDEN, alpha=1e-4, max_iter=MAX_EPOCHS,
                 random_state=0):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        targets = (np.asarray(y)[:, None] == self.classes_[None, :]).astype(float)
        settings = {**MLP_SETTINGS, "hidden_layer_sizes": self.hidden_layer_sizes,
                    "alpha": self.alpha, "max_iter": self.max_iter,
                    "random_state": self.random_state}
        self.mlp_ = MLPRegressor(**settings).fit(X, targets)
        return self

    def raw_output(self, X):
        out = self.mlp_.predict(X)
        return out.reshape(-1, len(self.classes_))

    def predict(self, X):
        return self.classes_[self.raw_output(X).argmax(axis=1)]

    def predict_proba(self, X):
        """Not real probabilities: clipped and rescaled so log loss can be computed."""
        p = np.clip(self.raw_output(X), EPS, None)
        return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- scoring --
# Plain callables (estimator, X, y) -> float; all are reported as computed.
def score_accuracy(est, X, y):
    return accuracy_score(y, est.predict(X))


def score_log_loss(est, X, y):
    return log_loss(y, est.predict_proba(X), labels=est.classes_)


def score_brier(est, X, y):
    p = est.predict_proba(X)
    onehot = (np.asarray(y)[:, None] == est.classes_[None, :]).astype(float)
    return np.mean(np.sum((p - onehot) ** 2, axis=1))


def score_outside_01(est, X, y):
    """Share of raw output values below 0 or above 1."""
    final = est[-1] if hasattr(est, "steps") else est
    if not isinstance(final, SquaredErrorMLP):
        return 0.0                      # softmax outputs are always inside [0, 1]
    Xt = est[:-1].transform(X) if hasattr(est, "steps") else X
    raw = final.raw_output(Xt)
    return float(np.mean((raw < 0) | (raw > 1)))


SCORING = {"acc": score_accuracy, "log_loss": score_log_loss,
           "brier": score_brier, "outside_01": score_outside_01}

MODELS = {
    "linear output + squared error": lambda: SquaredErrorMLP(),
    "softmax + cross-entropy": lambda: MLPClassifier(**MLP_SETTINGS),
    "logistic regression": lambda: LogisticRegression(C=1.0, max_iter=500),
}


def run(dataset, X, y):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    rows = []
    for name, make in MODELS.items():
        model = make_pipeline(make_tfidf(), make()) if dataset == "SAI" else make()
        start = time.time()
        s = cross_validate(model, X, y, cv=cv, scoring=SCORING, return_train_score=True,
                           n_jobs=-1, error_score="raise")
        row = {"dataset": dataset, "model": name,
               "train_acc": s["train_acc"].mean(), "val_acc": s["test_acc"].mean(),
               "val_std": s["test_acc"].std(),
               "train_log_loss": s["train_log_loss"].mean(),
               "val_log_loss": s["test_log_loss"].mean(),
               "val_brier": s["test_brier"].mean(),
               "val_outside_01": s["test_outside_01"].mean(),
               "seconds": round(time.time() - start, 1)}
        rows.append(row)
        print(f"{dataset:5s} | {name:30s} | train {row['train_acc']:.4f} | "
              f"val {row['val_acc']:.4f} ± {row['val_std']:.4f} | "
              f"val logloss {row['val_log_loss']:.3f} | brier {row['val_brier']:.3f} | "
              f"outputs outside [0,1] {row['val_outside_01']:.1%} | {row['seconds']} s",
              flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probability output experiment (3.1 d)")
    parser.add_argument("--data", nargs="+", choices=["MNIST", "SAI"], default=["MNIST", "SAI"])
    args = parser.parse_args()

    out_dir = ROOT / "results/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.data:
        if dataset == "MNIST":
            X = load_images(ROOT / "data/mnist/train-images-idx3-ubyte.gz") / 255.0
            y = load_labels(ROOT / "data/mnist/train-labels-idx1-ubyte.gz")
        else:
            texts, y, _ = load_sai(ROOT / "data/sai/train.csv")
            X = np.array(texts, dtype=object)
        rows = run(dataset, X, y)
        out_file = out_dir / f"probability_output_{dataset}.csv"
        pd.DataFrame(rows).to_csv(out_file, index=False)
        print("Saved:", out_file, flush=True)