"""Figures for the Question 3.1 report, made from the cross-validation results.

Reads every results/tables/experiments_cv_*.csv and loss_curves_*.csv
(quick test runs are skipped), and saves each figure as PNG (300 dpi, for
Word/Docs) and PDF (vector, for LaTeX) in results/figures/.

Run from anywhere:  python src/plot_results.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files only, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"

# ------------------------------------------------------------------ style --
# Dataset identity: blue = MNIST, dark gold = SAI (validated colorblind-safe pair).
COLOR = {"MNIST": "#015DBE", "SAI": "#B8860B"}
TRAIN_GRAY = "#9A9A9A"      # training curves are context, drawn in gray
REFERENCE_GRAY = "#7A7A7A"
# Depth L = 1, 2, 3, 5, 8 is ordered, so it uses one blue ramp, light -> dark.
DEPTH_RAMP = ["#8DB6E8", "#5A95DA", "#2F78CF", "#015DBE", "#013A78"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#8A8A8A",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print("Saved:", FIGURES / f"{name}.png", "(+ .pdf)")


# ------------------------------------------------------------------- data --
def load_tables():
    files = [f for f in sorted(TABLES.glob("experiments_cv*.csv")) if "_quick" not in f.name]
    curve_files = [f for f in sorted(TABLES.glob("loss_curves*.csv")) if "_quick" not in f.name]
    if not files:
        raise SystemExit(f"No results found in {TABLES}. Run shallow_deep.py first.")
    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    res = res.drop_duplicates(["experiment", "dataset", "config"], keep="last")
    curves = pd.concat([pd.read_csv(f) for f in curve_files], ignore_index=True) \
        if curve_files else pd.DataFrame()
    if not curves.empty:
        curves = curves.drop_duplicates(["experiment", "dataset", "config", "fold", "epoch"],
                                        keep="last")
    res["L"] = res["hidden_layers"].map(lambda s: len(eval(s)))
    res["units"] = res["hidden_layers"].map(lambda s: eval(s)[0])
    print("Loaded:", ", ".join(f.name for f in files))
    return res, curves


def present(res, experiment):
    """Datasets that have rows for an experiment, MNIST first."""
    ds = res.loc[res["experiment"] == experiment, "dataset"].unique()
    return [d for d in ("MNIST", "SAI") if d in ds]


def pct_axis(ax):
    """Percent tick labels; one decimal when the axis range is narrow, so labels never repeat."""
    lo, hi = ax.get_ylim()
    digits = 1 if hi - lo < 6 else 0
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.{digits}f}%"))


# ---------------------------------------------------------------- figures --
def fig_depth_accuracy(res):
    """Train vs validation accuracy against depth, one panel per dataset."""
    datasets = present(res, "depth")
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 2.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = res[(res.experiment == "depth") & (res.dataset == ds)].sort_values("L")
        ax.plot(d.L, d.train_acc * 100, color=TRAIN_GRAY, marker="o", mfc="white",
                mew=1.5, label="Train")
        ax.errorbar(d.L, d.val_acc * 100, yerr=d.val_std * 100, color=COLOR[ds], marker="o",
                    capsize=3, elinewidth=1, label="Validation (mean ± std)")
        ax.set_xticks(d.L)
        ax.set_xlabel("Number of hidden layers (L)")
        ax.set_title(ds)
        pct_axis(ax)
    axes[0][0].set_ylabel("Accuracy (5-fold CV)")
    axes[0][-1].legend(loc="center right")
    fig.tight_layout()
    save(fig, "fig1_depth_accuracy")


def fig_gap(res):
    """Train - validation gap against depth; both datasets share one axis (same unit)."""
    datasets = present(res, "depth")
    if not datasets:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for ds, marker in zip(datasets, ["o", "s"]):
        d = res[(res.experiment == "depth") & (res.dataset == ds)].sort_values("L")
        ax.plot(d.L, d.gap * 100, color=COLOR[ds], marker=marker, label=ds)
        last = d.iloc[-1]
        ax.annotate(ds, (last.L, last.gap * 100), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#222222")
    ax.set_xticks(sorted(res.loc[res.experiment == "depth", "L"].unique()))
    ax.set_xlabel("Number of hidden layers (L)")
    ax.set_ylabel("Train − validation accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} pp"))
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.8)   # room for the end labels
    ax.set_title("Generalization gap")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout()
    save(fig, "fig2_generalization_gap")


def fig_depth_logloss(res):
    """Train vs validation log loss against depth, one panel per dataset."""
    datasets = present(res, "depth")
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 2.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = res[(res.experiment == "depth") & (res.dataset == ds)].sort_values("L")
        ax.plot(d.L, d.train_log_loss, color=TRAIN_GRAY, marker="o", mfc="white", mew=1.5,
                label="Train")
        ax.plot(d.L, d.val_log_loss, color=COLOR[ds], marker="o", label="Validation")
        ax.set_xticks(d.L)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Number of hidden layers (L)")
        ax.set_title(ds)
    axes[0][0].set_ylabel("Log loss (lower is better)")
    axes[0][-1].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_depth_logloss")


def fig_width(res):
    """Accuracy against hidden units (1 hidden layer); tick labels also show parameters."""
    datasets = present(res, "width")
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 3.0), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        d = res[(res.experiment == "width") & (res.dataset == ds)].sort_values("units")
        ax.plot(d.units, d.train_acc * 100, color=TRAIN_GRAY, marker="o", mfc="white",
                mew=1.5, label="Train")
        ax.errorbar(d.units, d.val_acc * 100, yerr=d.val_std * 100, color=COLOR[ds],
                    marker="o", capsize=3, elinewidth=1, label="Validation (mean ± std)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(d.units)
        ax.set_xticklabels([f"{u}\n({p / 1e6:.2f}M)" if p >= 1e6 else f"{u}\n({p / 1e3:.0f}k)"
                            for u, p in zip(d.units, d.n_params)])
        ax.minorticks_off()
        ax.set_xlabel("Hidden units (parameters)")
        ax.set_title(ds)
        pct_axis(ax)
    axes[0][0].set_ylabel("Accuracy (5-fold CV)")
    axes[0][-1].legend(loc="center right")
    fig.tight_layout()
    save(fig, "fig4_width_accuracy")


def fig_regularization(res):
    """SAI: effect of each regularization setting, as dot plots (no zero baseline needed)."""
    d = res[(res.experiment == "reg") & (res.dataset == "SAI")]
    if d.empty:
        return
    d = d.iloc[::-1]  # first config on top
    y = np.arange(len(d))
    is_base = d.config.str.startswith("baseline").to_numpy()
    colors = np.where(is_base, REFERENCE_GRAY, COLOR["SAI"])
    panels = [("val_acc", "Validation accuracy", 100, "val_std"),
              ("gap", "Train − validation gap", 100, None),
              ("val_log_loss", "Validation log loss", 1, None)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, (col, title, scale, err) in zip(axes, panels):
        vals = d[col].to_numpy() * scale
        if err:
            ax.errorbar(vals, y, xerr=d[err].to_numpy() * scale, fmt="none",
                        ecolor="#BBBBBB", elinewidth=1, capsize=2, zorder=1)
        ax.scatter(vals, y, c=colors, s=40, zorder=2, edgecolors="white", linewidths=1)
        if is_base.any():
            ax.axvline(vals[is_base][0], color=REFERENCE_GRAY, linewidth=0.8, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        if scale == 100:
            unit = "%" if col == "val_acc" else " pp"
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _, unit=unit: f"{v:.0f}{unit}"))
        ax.margins(x=0.15)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(d.config)
    fig.suptitle("SAI, 1 hidden layer (128 units): regularization settings "
                 "(gray = baseline)", fontsize=10, y=0.98)
    fig.tight_layout()
    save(fig, "fig5_sai_regularization")


def fig_loss_curves(curves):
    """Mean training loss per epoch for each depth (log scale), one panel per dataset."""
    if curves.empty:
        return
    c = curves[curves.experiment == "depth"]
    datasets = [ds for ds in ("MNIST", "SAI") if ds in c.dataset.unique()]
    if not datasets:
        return
    order = sorted(c.config.unique(), key=lambda s: int(s.split("=")[1]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 2.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for color, cfg in zip(DEPTH_RAMP, order):
            g = c[(c.dataset == ds) & (c.config == cfg)].groupby("epoch")["loss"]
            n_folds = c.loc[(c.dataset == ds) & (c.config == cfg), "fold"].nunique()
            # Folds can stop early; average only epochs that every fold reached,
            # otherwise the last points would mix different folds.
            m = g.mean()[g.count() == n_folds]
            ax.plot(m.index, m.values, color=color, linewidth=1.8, label=cfg)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(ds)
    axes[0][0].set_ylabel("Training loss (mean of folds)")
    axes[0][-1].legend(title="Hidden layers", loc="upper right", title_fontsize=9)
    fig.tight_layout()
    save(fig, "fig6_training_loss_curves")


def depth_colors(depths):
    """Spread the blue ramp over however many depths were run (light = shallow)."""
    idx = np.linspace(0, len(DEPTH_RAMP) - 1, num=len(depths)).round().astype(int)
    return {d: DEPTH_RAMP[i] for d, i in zip(sorted(depths), idx)}


def fig_vanishing_gradient():
    """(c) Gradient size per hidden layer in epoch 1, and training loss per epoch,
    for ReLU vs sigmoid. One pair of figures per dataset that has results."""
    for data in ("MNIST", "SAI"):
        gfile = TABLES / f"vanishing_gradient_{data}_grads.csv"
        efile = TABLES / f"vanishing_gradient_{data}_epochs.csv"
        if not gfile.exists():
            continue
        g = pd.read_csv(gfile)
        acts = [a for a in ("relu", "sigmoid") if a in g.activation.unique()]
        colors = depth_colors(g.depth.unique())
        names = {"relu": "ReLU", "sigmoid": "Sigmoid"}

        # Gradient size per hidden layer, first epoch (before training adapts).
        g1 = g[(g.epoch == 1) & (~g.is_output)]
        fig, axes = plt.subplots(1, len(acts), figsize=(3.4 * len(acts), 2.9),
                                 sharey=True, squeeze=False)
        for ax, act in zip(axes[0], acts):
            for depth in sorted(g1.depth.unique()):
                d = g1[(g1.activation == act) & (g1.depth == depth)].sort_values("layer")
                ax.plot(d.layer, d.grad_mean_abs, color=colors[depth], marker="o",
                        markersize=4, linewidth=1.8, label=f"{depth} hidden layers")
            ax.set_yscale("log")
            ax.set_xlabel("Hidden layer (1 = closest to input)")
            ax.set_title(names[act])
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        axes[0][0].set_ylabel(r"Mean $|\partial E / \partial w|$, epoch 1")
        axes[0][-1].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"{data}: gradient size per layer (plain SGD)", fontsize=10, y=1.0)
        fig.tight_layout()
        save(fig, f"fig7_gradient_per_layer_{data}")

        if not efile.exists():
            continue
        e = pd.read_csv(efile)
        n_classes = 10 if data == "MNIST" else 3
        chance = np.log(n_classes)   # loss of a model that gives every class 1/C
        fig, axes = plt.subplots(1, len(acts), figsize=(3.4 * len(acts), 2.8),
                                 sharey=True, squeeze=False)
        for ax, act in zip(axes[0], acts):
            ax.axhline(chance, color=REFERENCE_GRAY, linewidth=0.8, zorder=0)
            stuck = []
            for depth in sorted(e.depth.unique()):
                d = e[(e.activation == act) & (e.depth == depth)].sort_values("epoch")
                ax.plot(d.epoch, d.train_loss, color=colors[depth], linewidth=1.8,
                        label=f"{depth} hidden layers")
                if d.train_loss.iloc[-1] > 0.95 * chance:
                    stuck.append(str(depth))
            # Lines that never leave the chance level overlap; name them in text.
            note = f"chance level, ln {n_classes} ≈ {chance:.2f}"
            if stuck:
                note += f"\n{', '.join(stuck)} layers stay here"
            ax.text(0.97, chance - 0.08 * chance, note, transform=ax.get_yaxis_transform(),
                    ha="right", va="top", fontsize=8, color="#444444")
            ax.set_ylim(0, chance * 1.12)
            ax.set_xlabel("Epoch")
            ax.set_title(names[act])
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        axes[0][0].set_ylabel("Training loss")
        axes[0][0].legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=8)
        fig.suptitle(f"{data}: training loss, ReLU vs sigmoid (plain SGD)", fontsize=10, y=1.0)
        fig.tight_layout()
        save(fig, f"fig8_training_loss_activation_{data}")


def fig_probability_output():
    """(d) Accuracy, log loss and Brier score for squared-error vs softmax output,
    with logistic regression as baseline. Rows = datasets, columns = metrics."""
    frames = [pd.read_csv(f) for f in sorted(TABLES.glob("probability_output_*.csv"))]
    if not frames:
        return
    p = pd.concat(frames, ignore_index=True)
    datasets = [d for d in ("MNIST", "SAI") if d in p.dataset.unique()]
    order = ["linear output + squared error", "softmax + cross-entropy", "logistic regression"]
    panels = [("val_acc", "Validation accuracy", True),
              ("val_log_loss", "Validation log loss", False),
              ("val_brier", "Validation Brier score", False)]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(7.4, 0.6 + 1.7 * len(datasets)),
                             sharey=True, squeeze=False)
    for r, ds in enumerate(datasets):
        d = p[p.dataset == ds].set_index("model").reindex(order).dropna(how="all").iloc[::-1]
        y = np.arange(len(d))
        for c, (col, title, is_pct) in enumerate(panels):
            ax = axes[r][c]
            vals = d[col].to_numpy() * (100 if is_pct else 1)
            if is_pct:
                ax.errorbar(vals, y, xerr=d["val_std"].to_numpy() * 100, fmt="none",
                            ecolor="#BBBBBB", elinewidth=1, capsize=2, zorder=1)
            ax.scatter(vals, y, color=COLOR[ds], s=40, zorder=2,
                       edgecolors="white", linewidths=1)
            ax.grid(axis="y", visible=False)
            ax.margins(x=0.2)
            if is_pct:
                ax.xaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            if r == 0:
                ax.set_title(title, fontsize=10)
        axes[r][0].set_yticks(y)
        axes[r][0].set_yticklabels(d.index)
        axes[r][0].set_ylabel(ds, fontweight="bold")
    fig.text(0.99, 0.0, "Log loss and Brier score: lower is better", ha="right",
             fontsize=8, color="#555555")
    fig.tight_layout()
    save(fig, "fig9_probability_output")


if __name__ == "__main__":
    fig_probability_output()
    fig_vanishing_gradient()
    results, loss_curves = load_tables()
    fig_depth_accuracy(results)
    fig_gap(results)
    fig_depth_logloss(results)
    fig_width(results)
    fig_regularization(results)
    fig_loss_curves(loss_curves)