"""Figure 4: the receptive-field story in one chart -- RQ1 (32^3 baseline) vs RQ3b (16^3 single
window, no retrain) vs RQ4 (retrained scale-matched ensemble), across all three regions/size bins."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"

BLUE = "#0072B2"     # RQ1 (32^3 baseline)
GREEN = "#009E73"    # RQ3b (16^3 single window)
VERMILLION = "#D55E00"  # RQ4 (scale-matched, retrained)

REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]


def load_scores(filename):
    with open(os.path.join(RESULTS_DIR, filename), newline="") as f:
        return list(csv.DictReader(f))


def main():
    rq1 = load_scores("rq1_localization_scores.csv")
    rq3b = load_scores("rq3b_scale16only_scores.csv")
    rq4 = load_scores("rq4_localization_scores.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    x = np.arange(len(BINS))
    width = 0.26

    for ax, region in zip(axes, REGIONS):
        rq1_means = [np.mean([float(r["dice"]) for r in rq1 if r["region"] == region and r["size_bin"] == b]) for b in BINS]
        rq3b_means = [np.mean([float(r["dice"]) for r in rq3b if r["region"] == region and r["size_bin"] == b]) for b in BINS]
        rq4_means = [np.mean([float(r["dice"]) for r in rq4 if r["region"] == region and r["size_bin"] == b]) for b in BINS]

        ax.bar(x - width, rq1_means, width, color=BLUE, label="RQ1: 32³ baseline")
        ax.bar(x, rq3b_means, width, color=GREEN, label="RQ3b: 16³ single window (no retrain)")
        ax.bar(x + width, rq4_means, width, color=VERMILLION, label="RQ4: scale-matched (retrained ensemble)")
        ax.set_xticks(x)
        ax.set_xticklabels(BINS)
        ax.set_title(region, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", axis="y", color="#dddddd", linewidth=0.6, zorder=0)

    axes[0].set_ylabel("Dice score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig4_scale_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
