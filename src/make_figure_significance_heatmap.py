"""Figure: heatmap of every paired significance test run in this project (rows = comparison vs.
RQ1 baseline, columns = region x size bin), color = direction (blue=better, red=worse) and
saturation = whether it survives full-family BH-FDR correction. Diagnostic summary of ~90+ tests
in one glance, replacing scattered text tables."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"
REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
COLUMNS = [f"{r} {b}" for r in REGIONS for b in BINS]

COMPARISONS = [
    ("RQ2", "rq2_localization_scores.csv"),
    ("RQ3 (multi-scale ensemble)", "rq3_localization_scores.csv"),
    ("RQ3b (16³)", "rq3b_scale16only_scores.csv"),
    ("RQ3c (12³)", "rq3c_window12_scores.csv"),
    ("RQ3c (8³)", "rq3c_window8_scores.csv"),
    ("RQ3c (6³)", "rq3c_window6_scores.csv"),
    ("RQ4", "rq4_localization_scores.csv"),
    ("RQ6", "rq6_localization_scores.csv"),
    ("RQ5 (naturalistic)", "rq5_localization_scores.csv"),
]


def load(fn):
    with open(os.path.join(RESULTS_DIR, fn), newline="") as f:
        return list(csv.DictReader(f))


def main():
    rq1 = load("rq1_localization_scores.csv")
    all_p, all_dir, cell_index = [], [], []

    for name, fname in COMPARISONS:
        other = load(fname)
        lookup = {(r["patient_id"], r["region"]): float(r["dice"]) for r in other}
        for region in REGIONS:
            for b in BINS:
                pairs = [
                    (float(r["dice"]), lookup[(r["patient_id"], r["region"])])
                    for r in rq1 if r["region"] == region and r["size_bin"] == b and (r["patient_id"], r["region"]) in lookup
                ]
                if len(pairs) < 5:
                    all_p.append(1.0)
                    all_dir.append(0)
                    cell_index.append((name, f"{region} {b}"))
                    continue
                av, bv = zip(*pairs)
                _, p = stats.wilcoxon(av, bv)
                direction = 1 if np.mean(bv) > np.mean(av) else -1
                all_p.append(p)
                all_dir.append(direction)
                cell_index.append((name, f"{region} {b}"))

    q_all = stats.false_discovery_control(all_p, method="bh")

    matrix = np.zeros((len(COMPARISONS), len(COLUMNS)))
    for (name, col), d, p, q in zip(cell_index, all_dir, all_p, q_all):
        row = [c[0] for c in COMPARISONS].index(name)
        colidx = COLUMNS.index(col)
        if q < 0.05:
            matrix[row, colidx] = d * 1.0  # strong color: significant after correction
        elif p < 0.05:
            matrix[row, colidx] = d * 0.45  # weak color: significant raw only
        else:
            matrix[row, colidx] = 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels(COLUMNS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(COMPARISONS)))
    ax.set_yticklabels([c[0] for c in COMPARISONS], fontsize=8.5)
    ax.set_title("Every comparison vs. RQ1 baseline (blue = better, red = worse; solid = survives BH-FDR, pale = raw p<0.05 only, white = n.s.)", fontsize=8.5)

    for i in range(len(COMPARISONS)):
        for j in range(len(COLUMNS)):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#cccccc", linewidth=0.5))

    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_significance_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
