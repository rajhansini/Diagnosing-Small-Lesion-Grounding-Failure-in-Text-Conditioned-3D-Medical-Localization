"""Generate report figures. Colorblind-safe Okabe-Ito hues, consistent mapping across panels,
log-scale volume axis (dynamic range spans 32mm^3 to 360,000mm^3), direct legends, no dual-axis."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"

BLUE = "#0072B2"    # model / RQ1 (Okabe-Ito)
ORANGE = "#D55E00"  # RQ2 (Okabe-Ito)
GRAY = "#888888"    # chance / reference baseline

REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]


def load_scores(filename):
    """Read a per-patient score CSV from results/ into a list of dict rows.

    Args:
        filename: basename of the CSV inside the results directory.

    Returns:
        List of dicts, one per (patient, region) row.
    """
    with open(os.path.join(RESULTS_DIR, filename), newline="") as f:
        return list(csv.DictReader(f))


def fig1_dice_vs_volume():
    """Draw Figure 1: per-patient Dice against true lesion volume, with the chance baseline overlaid."""
    rq1 = load_scores("rq1_localization_scores.csv")
    chance = load_scores("chance_baseline_scores.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, region in zip(axes, REGIONS):
        rq1_r = [r for r in rq1 if r["region"] == region]
        chance_r = [r for r in chance if r["region"] == region]
        vols = np.array([float(r["true_volume_mm3"]) for r in rq1_r])
        dices = np.array([float(r["dice"]) for r in rq1_r])
        chance_dices = np.array([float(r["dice"]) for r in chance_r])
        rho, _ = stats.spearmanr(np.log10(vols), dices)

        ax.scatter(vols, chance_dices, s=18, color=GRAY, alpha=0.6, label="chance (random heatmap)", zorder=2)
        ax.scatter(vols, dices, s=18, color=BLUE, alpha=0.75, label="model (RQ1)", zorder=3)
        ax.set_xscale("log")
        ax.set_title(f"{region}  (Spearman ρ={rho:.2f})", fontsize=11)
        ax.set_xlabel("true lesion volume (mm³, log scale)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", axis="both", color="#dddddd", linewidth=0.6, zorder=0)

    axes[0].set_ylabel("Dice score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
    fig.suptitle("")
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig1_dice_vs_volume.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def fig2_rq1_vs_rq2_bars():
    """Draw Figure 2: RQ1 vs RQ2 mean Dice by region and size bin."""
    rq1 = load_scores("rq1_localization_scores.csv")
    rq2 = load_scores("rq2_localization_scores.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    x = np.arange(len(BINS))
    width = 0.35

    for ax, region in zip(axes, REGIONS):
        rq1_means = [np.mean([float(r["dice"]) for r in rq1 if r["region"] == region and r["size_bin"] == b]) for b in BINS]
        rq1_stds = [np.std([float(r["dice"]) for r in rq1 if r["region"] == region and r["size_bin"] == b]) for b in BINS]
        rq2_means = [np.mean([float(r["dice"]) for r in rq2 if r["region"] == region and r["size_bin"] == b]) for b in BINS]
        rq2_stds = [np.std([float(r["dice"]) for r in rq2 if r["region"] == region and r["size_bin"] == b]) for b in BINS]

        ax.bar(x - width / 2, rq1_means, width, yerr=rq1_stds, capsize=3, color=BLUE, label="RQ1 (baseline)")
        ax.bar(x + width / 2, rq2_means, width, yerr=rq2_stds, capsize=3, color=ORANGE, label="RQ2 (size-conditioned)")
        ax.set_xticks(x)
        ax.set_xticklabels(BINS)
        ax.set_title(region, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", axis="y", color="#dddddd", linewidth=0.6, zorder=0)

    axes[0].set_ylabel("Dice score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2, frameon=False)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig2_rq1_vs_rq2.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig1_dice_vs_volume()
    fig2_rq1_vs_rq2_bars()
