"""Charts for the final report. Every number here is computed upstream."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as cfg

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 10, "legend.frameon": False, "lines.linewidth": 2,
})


def pr_curve_chart(curves, threshold, path):
    """Precision-recall on the left, and the recall bought per unit of FPR on the right.

    `curves` maps a split label to its account roll-up.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {"Validation": BLUE, "Test": ORANGE}

    for label, accounts in curves.items():
        y, s = accounts["is_mule"], accounts["risk_score"]
        flagged = s >= threshold
        precision, recall, _ = precision_recall_curve(y, s)
        left.plot(recall, precision, color=colors[label], label=label)
        point_recall = (flagged & (y == 1)).sum() / (y == 1).sum()
        point_precision = (flagged & (y == 1)).sum() / max(flagged.sum(), 1)
        left.scatter([point_recall], [point_precision], s=70, color=colors[label],
                     edgecolor=SURFACE, linewidth=2, zorder=3)

        fpr, tpr, _ = roc_curve(y, s)
        keep = fpr > 0
        right.plot(fpr[keep] * 100, tpr[keep], color=colors[label], label=label)
        point_fpr = (flagged & (y == 0)).sum() / (y == 0).sum()
        right.scatter([max(point_fpr * 100, 1e-4)], [point_recall], s=70, color=colors[label],
                      edgecolor=SURFACE, linewidth=2, zorder=3)

    left.set(title="Precision-recall, per account (dots = operating point)",
             xlabel="Recall (share of mules caught)", ylabel="Precision (share of flags correct)",
             xlim=(0, 1.01), ylim=(0, 1.03))
    left.legend(loc="lower left")

    right.axvline(cfg.MAX_FPR * 100, color=MUTED, linestyle="--", linewidth=1.2)
    right.annotate(f"{cfg.MAX_FPR:.1%} FPR cap", (cfg.MAX_FPR * 100, 0.04), xytext=(6, 0),
                   textcoords="offset points", color=MUTED)
    right.set_xscale("log")
    right.set(title="Recall against false positive rate", xlabel="False positive rate (%, log)",
              ylabel="Recall", ylim=(0, 1.03))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def ranked_bars(labels, values, title, xlabel, path, formatter="{:.1%}", xlim=None):
    """One hue, largest first, each bar labelled with its value."""
    order = np.argsort(values)
    labels, values = np.array(labels)[order], np.array(values)[order]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(values) + 1.5))
    ax.barh(labels, values, color=BLUE, height=0.62)
    span = xlim[1] if xlim else max(values)
    for i, value in enumerate(values):
        ax.text(value + span * 0.012, i, formatter.format(value), va="center", color=INK)
    ax.set(title=title, xlabel=xlabel)
    ax.set_xlim(*(xlim or (0, span * 1.14)))
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fraud_rate_chart(table, path):
    """Daily fraud rate on the scored population, with the splits marked."""
    day = (table["step"].to_numpy().astype(int) - 1) // cfg.HOURS_PER_DAY
    daily = table.assign(day=day).groupby("day")["is_fraud"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(daily.index, daily["mean"] * 100, color=BLUE, marker="o", markersize=4)
    ax.set_yscale("log")
    ax.set(title="Fraud rate per day: stable until legitimate volume collapses on day 17",
           xlabel="Day of simulation", ylabel="Fraud rate (%, log)")

    edges = table.groupby("split")["step"].agg(["min", "max"])
    for split in ("train", "val", "test", "stress"):
        low = (edges.loc[split, "min"] - 1) / cfg.HOURS_PER_DAY - 0.5
        high = edges.loc[split, "max"] / cfg.HOURS_PER_DAY - 0.5
        ax.axvline(high, color=MUTED, linewidth=0.8, linestyle="--")
        ax.text((low + high) / 2, 0.92, split, color=MUTED, ha="center",
                transform=ax.get_xaxis_transform())
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def feedback_chart(frames, summaries, first_step, period_hours, path):
    """Recall per period in each dry-run scenario, with model promotions marked."""
    systems = {"self_learning": ("Self-learning", BLUE), "static_model": ("Static model", ORANGE),
               "one_line_rule": ("One-line rule", AQUA)}
    titles = {"control": "Control: PaySim unchanged",
              "evasive": "Evasive: mules stop emptying accounts"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    for ax, (name, frame) in zip(axes, frames.items()):
        period = ((frame["end"] - first_step) // period_hours).rename("period")
        totals = frame.groupby([period, "system"])[["tp", "fn"]].sum()
        recall = (totals["tp"] / (totals["tp"] + totals["fn"])).unstack("system")
        days = (recall.index + 1) * period_hours / cfg.HOURS_PER_DAY
        for system, (label, color) in systems.items():
            ax.plot(days, recall[system], color=color, marker="o", markersize=5, label=label)
        promotions = [r["step"] for r in summaries[name]["retrains"] if r["promoted"]]
        for i, step in enumerate(promotions):
            ax.axvline((step - first_step + 1) / cfg.HOURS_PER_DAY, color=MUTED, linestyle="--",
                       linewidth=1, label="New model promoted" if i == 0 else None)
        ax.set(title=titles[name], xlabel="Days into the replay", ylim=(0, 1.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    axes[0].set_ylabel("Recall, per account")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
