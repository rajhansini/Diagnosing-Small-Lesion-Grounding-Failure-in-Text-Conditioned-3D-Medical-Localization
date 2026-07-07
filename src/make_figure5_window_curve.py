"""Figure 5: Dice vs. sliding-window size (32/16/12/8), one line per region, showing the trend
continues (no reversal yet) as the receptive field shrinks further below the original 32^3 baseline."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"

REGION_COLORS = {"ET": "#0072B2", "TC": "#D55E00", "WT": "#009E73"}  # Okabe-Ito, consistent identity per region
WINDOW_SIZES = [32, 16, 12, 8]
FILES = {
    32: "rq1_localization_scores.csv",
    16: "rq3b_scale16only_scores.csv",
    12: "rq3c_window12_scores.csv",
    8: "rq3c_window8_scores.csv",
}


def load_scores(filename):
    with open(os.path.join(RESULTS_DIR, filename), newline="") as f:
        return list(csv.DictReader(f))


def main():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for region, color in REGION_COLORS.items():
        means = []
        for ws in WINDOW_SIZES:
            rows = load_scores(FILES[ws])
            vals = [float(r["dice"]) for r in rows if r["region"] == region]
            means.append(np.mean(vals))
        ax.plot(WINDOW_SIZES, means, marker="o", color=color, label=region, linewidth=2, markersize=6)

    ax.set_xscale("log", base=2)
    ax.set_xticks(WINDOW_SIZES)
    ax.set_xticklabels([str(w) for w in WINDOW_SIZES])
    ax.invert_xaxis()
    ax.set_xlabel("sliding-window size (voxels, log scale, smaller → right)")
    ax.set_ylabel("mean Dice (all size bins pooled)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", axis="both", color="#dddddd", linewidth=0.6, zorder=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig5_window_curve.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
