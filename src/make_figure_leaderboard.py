"""Figure: full leaderboard across every method tried in this project (RQ1 baseline, RQ2 size
prompting, RQ3c-8cubed best single window, RQ4 scale-matched, RQ6 uniformity-fixed), one panel
per region, one bar per method per size bin."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"

REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]

METHODS = [
    ("RQ1\n(32³ baseline)", "rq1_localization_scores.csv", "#0072B2"),
    ("RQ2\n(size prompt)", "rq2_localization_scores.csv", "#D55E00"),
    ("RQ3c\n(8³ window)", "rq3c_window8_scores.csv", "#009E73"),
    ("RQ4\n(scale-matched)", "rq4_localization_scores.csv", "#CC79A7"),
    ("RQ6\n(hub-fixed)", "rq6_localization_scores.csv", "#E69F00"),
]


def load_scores(filename):
    """Read a per-patient score CSV from results/ into a list of dict rows.

    Args:
        filename: basename of the CSV inside the results directory.

    Returns:
        List of dicts, one per (patient, region) row.
    """
    with open(os.path.join(RESULTS_DIR, filename), newline="") as f:
        return list(csv.DictReader(f))


def main():
    """Draw the leaderboard bar chart of every method's mean Dice, one panel per region."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    x = np.arange(len(BINS))
    width = 0.16

    for ax, region in zip(axes, REGIONS):
        for i, (label, fname, color) in enumerate(METHODS):
            rows = load_scores(fname)
            means = [np.mean([float(r["dice"]) for r in rows if r["region"] == region and r["size_bin"] == b]) for b in BINS]
            offset = (i - (len(METHODS) - 1) / 2) * width
            ax.bar(x + offset, means, width, color=color, label=label if region == "ET" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(BINS)
        ax.set_title(region, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", axis="y", color="#dddddd", linewidth=0.6, zorder=0)

    axes[0].set_ylabel("Dice score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=5, frameon=False, fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_leaderboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
