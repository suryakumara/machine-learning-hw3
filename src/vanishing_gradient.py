# type: ignore
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch 
from sklearn.model_selection import train_test_split
from torch import nn 

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mnist_loader import load_images, load_labels  # noqa: E402
from sai_loader import load_sai, make_tfidf  # noqa: E402

parser = argparse.ArgumentParser(description="Vanishing gradient experiment (Question 3.1 c)")
parser.add_argument("--data", choices=["MNIST", "SAI"], default="MNIST")
parser.add_argument("--depths", nargs="+", type=int, default=[3, 5, 8, 15],
                    help="numbers of hidden layers to compare")
parser.add_argument("--activations", nargs="+", choices=["relu", "sigmoid"],
                    default=["relu", "sigmoid"])
parser.add_argument("--units", type=int, default=128, help="neurons per hidden layer")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.05, help="SGD learning rate")
parser.add_argument("--batch", type=int, default=128, help="mini-batch size")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)


# ------------------------------------------------------------------- data --
def load_data():
    """Return (X_train, y_train, X_val, y_val, n_inputs, n_classes).
    X is a dense NumPy array (MNIST) or a sparse matrix (SAI TF-IDF)."""
    if args.data == "MNIST":
        X = (load_images(ROOT / "data/mnist/train-images-idx3-ubyte.gz") / 255.0).astype(np.float32)
        y = load_labels(ROOT / "data/mnist/train-labels-idx1-ubyte.gz").astype(np.int64)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=args.seed)
        return Xtr, ytr, Xva, yva, X.shape[1], 10
    texts, y, _ = load_sai(ROOT / "data/sai/train.csv")
    ttr, tva, ytr, yva = train_test_split(np.array(texts, dtype=object), y.astype(np.int64),
                                          test_size=0.2, stratify=y, random_state=args.seed)
    vec = make_tfidf()                       # fitted on the training split only
    Xtr = vec.fit_transform(ttr).astype(np.float32)
    Xva = vec.transform(tva).astype(np.float32)
    return Xtr, ytr, Xva, yva, Xtr.shape[1], 3


def to_tensor(X):
    """Dense float tensor on the device; sparse TF-IDF rows are densified per batch."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return torch.from_numpy(np.asarray(X)).to(device)


def batches(n, shuffle, generator):
    order = torch.randperm(n, generator=generator).numpy() if shuffle else np.arange(n)
    for start in range(0, n, args.batch):
        yield order[start:start + args.batch]


# ------------------------------------------------------------------ model --
def make_model(n_in, n_out, depth, activation):
    act = nn.ReLU if activation == "relu" else nn.Sigmoid
    layers, width = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(width, args.units), act()]
        width = args.units
    layers.append(nn.Linear(width, n_out))   # output layer; softmax is inside the loss
    return nn.Sequential(*layers).to(device)


def evaluate(model, X, y, loss_fn):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for idx in batches(len(y), shuffle=False, generator=None):
            xb, yb = to_tensor(X[idx]), torch.from_numpy(y[idx]).to(device)
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(y), correct / len(y)


def train_one(Xtr, ytr, Xva, yva, n_in, n_out, depth, activation):
    torch.manual_seed(args.seed)
    model = make_model(n_in, n_out, depth, activation)
    linears = [m for m in model if isinstance(m, nn.Linear)]   # depth hidden + 1 output
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # plain SGD, no momentum
    loss_fn = nn.CrossEntropyLoss()                               # softmax + log loss
    gen = torch.Generator().manual_seed(args.seed)

    grad_rows, epoch_rows = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        grad_sum = np.zeros(len(linears))
        loss_sum, n_batches = 0.0, 0
        for idx in batches(len(ytr), shuffle=True, generator=gen):
            xb, yb = to_tensor(Xtr[idx]), torch.from_numpy(ytr[idx]).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            # Mean |gradient| per weight in every layer, measured before the update.
            grad_sum += [lin.weight.grad.abs().mean().item() for lin in linears]
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1

        grads = grad_sum / n_batches
        val_loss, val_acc = evaluate(model, Xva, yva, loss_fn)
        for layer, g in enumerate(grads, start=1):
            grad_rows.append({"data": args.data, "activation": activation, "depth": depth,
                              "epoch": epoch, "layer": layer, "n_layers": len(linears),
                              "is_output": layer == len(linears), "grad_mean_abs": g})
        epoch_rows.append({"data": args.data, "activation": activation, "depth": depth,
                           "epoch": epoch, "train_loss": loss_sum / n_batches,
                           "val_loss": val_loss, "val_acc": val_acc,
                           "first_to_last_hidden_ratio": grads[0] / grads[depth - 1]})
        print(f"{activation:7s} | depth {depth:2d} | epoch {epoch:2d} | "
              f"train loss {loss_sum / n_batches:.4f} | val acc {val_acc:.4f} | "
              f"|grad| layer 1 {grads[0]:.2e}, last hidden {grads[depth - 1]:.2e} "
              f"(ratio {grads[0] / grads[depth - 1]:.3g})", flush=True)
    return grad_rows, epoch_rows


if __name__ == "__main__":
    Xtr, ytr, Xva, yva, n_in, n_out = load_data()
    print(f"{args.data}: {len(ytr)} train / {len(yva)} validation, {n_in} inputs, "
          f"{n_out} classes", flush=True)

    out_dir = ROOT / "results/tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    grads_file = out_dir / f"vanishing_gradient_{args.data}_grads.csv"
    epochs_file = out_dir / f"vanishing_gradient_{args.data}_epochs.csv"

    all_grads, all_epochs = [], []
    for activation in args.activations:
        for depth in args.depths:
            start = time.time()
            g, e = train_one(Xtr, ytr, Xva, yva, n_in, n_out, depth, activation)
            all_grads += g
            all_epochs += e
            # Save after every network, so finished results are kept.
            pd.DataFrame(all_grads).to_csv(grads_file, index=False)
            pd.DataFrame(all_epochs).to_csv(epochs_file, index=False)
            print(f"--- {activation} depth {depth} done in {time.time() - start:.1f} s ---",
                  flush=True)
    print("Saved:", grads_file)
    print("Saved:", epochs_file)