"""Statistical validation: Spearman correlation (Dice vs continuous lesion size), paired Wilcoxon
signed-rank tests (RQ1 vs RQ2 per size bin), and lift-over-chance (model Dice / random-heatmap Dice)
per size bin -- this last one controls for Dice's inherent geometric sensitivity to small structures."""
import csv
import os

import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"


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
    """Report Spearman correlation of Dice against lesion volume and lift over the chance baseline."""
    rq1 = load_scores("rq1_localization_scores.csv")
    rq2 = load_scores("rq2_localization_scores.csv")
    chance = load_scores("chance_baseline_scores.csv")

    print("=== Spearman correlation: Dice vs log10(true lesion volume), RQ1 model vs chance ===")
    for region in ["ET", "TC", "WT"]:
        rq1_region = [r for r in rq1 if r["region"] == region]
        chance_region = [r for r in chance if r["region"] == region]

        vols = np.log10([float(r["true_volume_mm3"]) for r in rq1_region])
        dices = [float(r["dice"]) for r in rq1_region]
        rho, p = stats.spearmanr(vols, dices)

        chance_dices = [float(r["dice"]) for r in chance_region]
        rho_c, p_c = stats.spearmanr(vols, chance_dices)

        print(f"{region}: n={len(rq1_region)} | model rho={rho:.3f} (p={p:.2e}) | chance rho={rho_c:.3f} (p={p_c:.2e})")

    print("\n=== Lift over chance (model Dice / chance Dice), per region x size bin ===")
    for region in ["ET", "TC", "WT"]:
        for b in ["small", "medium", "large"]:
            m = np.mean([float(r["dice"]) for r in rq1 if r["region"] == region and r["size_bin"] == b])
            c = np.mean([float(r["dice"]) for r in chance if r["region"] == region and r["size_bin"] == b])
            print(f"{region} / {b}: model={m:.4f}, chance={c:.4f}, lift={m / c:.2f}x")

    print("\n=== Paired Wilcoxon signed-rank test: RQ1 vs RQ2 Dice, per region x size bin ===")
    rq2_lookup = {(r["patient_id"], r["region"]): float(r["dice"]) for r in rq2}
    for region in ["ET", "TC", "WT"]:
        for b in ["small", "medium", "large"]:
            pairs = [
                (float(r["dice"]), rq2_lookup[(r["patient_id"], r["region"])])
                for r in rq1 if r["region"] == region and r["size_bin"] == b and (r["patient_id"], r["region"]) in rq2_lookup
            ]
            if len(pairs) < 5:
                continue
            rq1_vals, rq2_vals = zip(*pairs)
            stat, p = stats.wilcoxon(rq1_vals, rq2_vals)
            direction = "RQ2 > RQ1" if np.mean(rq2_vals) > np.mean(rq1_vals) else "RQ2 <= RQ1"
            sig = "*" if p < 0.05 else "n.s."
            print(f"{region} / {b}: n={len(pairs)}, {direction}, p={p:.4f} [{sig}]")


if __name__ == "__main__":
    main()
