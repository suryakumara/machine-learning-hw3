"""Final evaluation on the MNIST test set (10,000 images).

Use this ONCE, after choosing the best configuration from the cross-validation
results. It trains that configuration on all 60,000 training images and
reports accuracy and log loss on the test set.

Examples (run from anywhere):
  python src/final_test.py --hidden 128
  python src/final_test.py --hidden 128 128 --alpha 1e-3
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mnist_loader import load_images, load_labels  # noqa: E402

parser = argparse.ArgumentParser(description="Final MNIST test-set evaluation")
parser.add_argument("--hidden", nargs="+", type=int, required=True,
                    help="neurons per hidden layer, e.g. --hidden 128 128")
parser.add_argument("--alpha", type=float, default=1e-4, help="L2 penalty (weight decay)")
parser.add_argument("--epochs", type=int, default=20, help="maximum number of epochs")
args = parser.parse_args()

X_train = load_images(ROOT / "data/mnist/train-images-idx3-ubyte.gz") / 255.0
y_train = load_labels(ROOT / "data/mnist/train-labels-idx1-ubyte.gz")
X_test = load_images(ROOT / "data/mnist/t10k-images-idx3-ubyte.gz") / 255.0
y_test = load_labels(ROOT / "data/mnist/t10k-labels-idx1-ubyte.gz")

# Same settings as in shallow_deep.py, so the result matches the chosen configuration.
model = MLPClassifier(
    hidden_layer_sizes=tuple(args.hidden),
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    batch_size=128,
    max_iter=args.epochs,
    alpha=args.alpha,
    random_state=0,
)

start = time.time()
model.fit(X_train, y_train)          # all 60,000 training images
seconds = round(time.time() - start, 1)

row = {
    "hidden_layers": str(tuple(args.hidden)),
    "alpha": args.alpha,
    "epochs_run": model.n_iter_,
    "train_acc": accuracy_score(y_train, model.predict(X_train)),
    "test_acc": accuracy_score(y_test, model.predict(X_test)),
    "test_log_loss": log_loss(y_test, model.predict_proba(X_test)),
    "seconds": seconds,
}
print(f"MNIST test | hidden {row['hidden_layers']} | alpha {args.alpha} | "
      f"train {row['train_acc']:.4f} | test {row['test_acc']:.4f} | "
      f"test logloss {row['test_log_loss']:.3f} | {seconds} s")

out_file = ROOT / "results/tables/final_test_mnist.csv"
out_file.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([row]).to_csv(out_file, mode="a", header=not out_file.exists(), index=False)
print("Appended to:", out_file)