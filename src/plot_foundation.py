
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLOR = {"MNIST": "#015DBE", "SAI": "#B8860B"}   # same colors as the other figures
GRAY = "#9A9A9A"
INK = "#222222"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8A8A8A", "axes.linewidth": 0.6,
    "xtick.color": "#444444", "ytick.color": "#444444",
})

# Labels shown on the y-axis, in the order they are drawn (top to bottom).
# (file, column value, label, is_foundation_model)
ROWS = {
    "MNIST": [
        ("prob", "logistic regression", "Logistic regression", False),
        ("prob", "softmax + cross-entropy", "MLP 1 × 128", False),
        ("fm", "dinov2", "DINOv2 (frozen) + LR", True),
    ],
    "SAI": [
        ("fm", "minilm", "MiniLM (frozen) + LR", True),
        ("prob", "logistic regression", "TF-IDF + logistic regression", False),
        ("prob", "softmax + cross-entropy", "TF-IDF + MLP 1 × 128", False),
        ("fm", "distilbert", "DistilBERT (fine-tuned)", True),
    ],
}


def load(dataset):
    """Return a list of (label, mean %, std %, is_foundation) for one dataset."""
    prob = pd.read_csv(TABLES / f"probability_output_{dataset}.csv").set_index("model")
    fm = pd.read_csv(TABLES / "foundation_model.csv")
    fm = fm[fm["dataset"] == dataset].set_index("method")
    rows = []
    for source, key, label, is_fm in ROWS[dataset]:
        table = prob if source == "prob" else fm
        if key not in table.index:
            print(f"  skipped {dataset} / {key}: not found in the table")
            continue
        r = table.loc[key]
        rows.append((label, 100 * r["val_acc"], 100 * r["val_std"], is_fm))
    return rows


fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
for ax, dataset in zip(axes, ["MNIST", "SAI"]):
    rows = load(dataset)
    ys = range(len(rows))[::-1]                      # first row at the top
    for y, (label, mean, std, is_fm) in zip(ys, rows):
        color = COLOR[dataset] if is_fm else GRAY
        ax.errorbar(mean, y, xerr=std, fmt="o", markersize=7, color=color,
                    ecolor=color, elinewidth=1.2, capsize=3, zorder=3)
        ax.annotate(f"{mean + 1e-9:.2f}", (mean, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows])
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_fontweight("bold" if row[3] else "normal")
    lo = min(r[1] - r[2] for r in rows)
    hi = max(r[1] + r[2] for r in rows)
    pad = 0.12 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("Validation accuracy (%), 5-fold CV")
    ax.set_title(dataset, fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.set_axisbelow(True)

fig.text(0.5, -0.04, "Bold and colored: foundation models. Gray: earlier models. "
         "Bars show ± one standard deviation over the 5 folds.",
         ha="center", fontsize=8, color="#555555")
fig.tight_layout(w_pad=3)
for ext in ("png", "pdf"):
    fig.savefig(FIGURES / f"fig10_foundation_models.{ext}", dpi=300,
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved:", FIGURES / "fig10_foundation_models.png")
