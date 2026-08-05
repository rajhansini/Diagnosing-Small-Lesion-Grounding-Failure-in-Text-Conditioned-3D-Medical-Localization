"""RQ11 analysis: decomposing the reported size collapse into grounding failure vs. threshold miscalibration.

Consumes results/rq11_threshold_confound_scores.csv and answers four questions the raw evaluation
log cannot:

1. How badly is Otsu miscalibrated, in absolute terms and as a function of lesion size?
2. How much Dice does the Otsu protocol cost, paired per patient, against an oracle-volume
   threshold that removes calibration from the problem entirely?
3. Does the size collapse survive once thresholding is controlled for -- and by how much does its
   magnitude change?
4. Is the pointing game (peak-response voxel inside the true mask) actually size-independent?
   It has no Dice-style geometric penalty, but a larger target is still easier to hit by chance,
   so the hit rate is reported against its own per-bin chance baseline with an exact binomial test
   rather than presented as if chance were zero.
"""
import csv
import os

import numpy as np
from scipy import stats

RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
SCORES_CSV = os.path.join(RESULTS_DIR, "rq11_threshold_confound_scores.csv")
REGIONS = ["ET", "TC", "WT"]
BINS = ["small", "medium", "large"]
METHODS = ["otsu", "pct90", "pct95", "pct99", "oracle_volume"]

GRID_VOXELS = 128 ** 3
VOXEL_VOLUME_MM3 = 4.2572021484375
BRAIN_BOX_MM3 = GRID_VOXELS * VOXEL_VOLUME_MM3


def load():
    """Read the RQ11 multi-threshold score CSV.

    Returns:
        List of dicts, one per (patient, region, threshold_method) row.
    """
    with open(SCORES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("true_volume_mm3", "gt_volume_mm3_resized", "pred_volume_mm3", "dice", "iou", "argmax_dist_mm"):
            r[k] = float(r[k])
        for k in ("gt_voxels_resized", "pred_voxels", "argmax_hit"):
            r[k] = int(r[k])
    return rows


def sel(rows, **kw):
    """Extract the float Dice values from a list of already-filtered rows.

    Args:
        rows: rows to pull the "dice" field from.

    Returns:
        List of floats.
    """
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def main():
    """Decompose the measured size collapse into grounding failure and binarization failure."""
    rows = load()
    print("=" * 100)
    print("RQ11: is the size-dependent Dice collapse grounding failure, or threshold miscalibration?")
    print(f"imaged volume = {GRID_VOXELS:,} voxels = {BRAIN_BOX_MM3:,.0f} mm^3")
    print("=" * 100)

    # ---- 1. How miscalibrated is Otsu? ----
    print("\n[1] OTSU CALIBRATION: predicted mask size vs. ground-truth lesion size")
    print(f"{'region':<5}{'bin':<8}{'n':>4}{'GT mm^3':>12}{'Otsu pred mm^3':>16}{'over-pred':>11}{'% of volume':>13}")
    for region in REGIONS:
        for b in BINS:
            s = sel(rows, region=region, size_bin=b, threshold_method="otsu")
            if not s:
                continue
            gt = np.mean([r["gt_volume_mm3_resized"] for r in s])
            pred = np.mean([r["pred_volume_mm3"] for r in s])
            print(f"{region:<5}{b:<8}{len(s):>4}{gt:>12,.0f}{pred:>16,.0f}{pred / gt:>10.1f}x{100 * pred / BRAIN_BOX_MM3:>12.1f}%")

    print("\n  Otsu's predicted volume vs. true lesion volume (Spearman, all bins pooled per region):")
    for region in REGIONS:
        s = sel(rows, region=region, threshold_method="otsu")
        rho, p = stats.spearmanr([r["true_volume_mm3"] for r in s], [r["pred_volume_mm3"] for r in s])
        print(f"    {region}: rho={rho:+.3f} (p={p:.2e}) over n={len(s)}")
    print("  A calibrated predictor would show rho close to +1. Negative rho means Otsu emits a")
    print("  LARGER mask for a SMALLER lesion, which mechanically inflates the measured size collapse.")

    # ---- 2. Paired cost of the thresholding step ----
    print("\n[2] PAIRED COST OF THE OTSU STEP (same patient, same heatmap, different binarization)")
    print(f"{'region':<5}{'bin':<8}{'n':>4}{'Otsu':>9}{'oracle':>9}{'delta':>9}{'x gain':>8}{'p (Wilcoxon)':>14}")
    for region in REGIONS:
        for b in BINS:
            otsu = {r["patient_id"]: r["dice"] for r in sel(rows, region=region, size_bin=b, threshold_method="otsu")}
            oracle = {r["patient_id"]: r["dice"] for r in sel(rows, region=region, size_bin=b, threshold_method="oracle_volume")}
            shared = sorted(set(otsu) & set(oracle))
            if len(shared) < 5:
                continue
            a = np.array([otsu[p] for p in shared])
            c = np.array([oracle[p] for p in shared])
            p = stats.wilcoxon(a, c).pvalue
            gain = c.mean() / a.mean() if a.mean() > 0 else float("nan")
            print(f"{region:<5}{b:<8}{len(shared):>4}{a.mean():>9.4f}{c.mean():>9.4f}{c.mean() - a.mean():>+9.4f}{gain:>7.1f}x{p:>14.2e}")

    # ---- 3. Does the collapse survive? ----
    print("\n[3] DOES THE SIZE COLLAPSE SURVIVE EACH BINARIZATION RULE?")
    print("    (large/small Dice ratio, and Spearman of Dice against true lesion volume)")
    print(f"{'region':<5}{'method':<15}{'small':>9}{'medium':>9}{'large':>9}{'L/S':>8}{'rho(dice,vol)':>15}")
    for region in REGIONS:
        for method in METHODS:
            means = {}
            for b in BINS:
                vals = [r["dice"] for r in sel(rows, region=region, size_bin=b, threshold_method=method)]
                means[b] = np.mean(vals) if vals else float("nan")
            s = sel(rows, region=region, threshold_method=method)
            rho, _ = stats.spearmanr([r["true_volume_mm3"] for r in s], [r["dice"] for r in s])
            ratio = means["large"] / means["small"] if means["small"] > 0 else float("nan")
            print(f"{region:<5}{method:<15}{means['small']:>9.4f}{means['medium']:>9.4f}{means['large']:>9.4f}"
                  f"{ratio:>7.1f}x{rho:>15.3f}")

    # ---- 4. Pointing game against its own chance baseline ----
    print("\n[4] POINTING GAME vs. CHANCE (peak-response voxel inside the true mask)")
    print("    chance = E[hit] for a uniformly random voxel = mean(GT voxels / total voxels).")
    print("    p is an exact one-sided binomial test of hits > chance.")
    print(f"{'region':<5}{'bin':<8}{'n':>4}{'hits':>6}{'hit rate':>10}{'chance':>9}{'lift':>8}{'p':>11}"
          f"{'mean dist':>11}{'median':>9}")
    for region in REGIONS:
        for b in BINS:
            s = sel(rows, region=region, size_bin=b, threshold_method="otsu")  # metrics are per-heatmap
            if not s:
                continue
            hits = sum(r["argmax_hit"] for r in s)
            n = len(s)
            chance = float(np.mean([r["gt_voxels_resized"] / GRID_VOXELS for r in s]))
            p = stats.binomtest(hits, n, chance, alternative="greater").pvalue
            dists = [r["argmax_dist_mm"] for r in s if not np.isnan(r["argmax_dist_mm"])]
            lift = (hits / n) / chance if chance > 0 else float("nan")
            print(f"{region:<5}{b:<8}{n:>4}{hits:>6}{hits / n:>10.3f}{chance:>9.4f}{lift:>7.1f}x{p:>11.2e}"
                  f"{np.mean(dists):>10.1f}m{np.median(dists):>8.1f}m")

    print("\n  Unlike Dice, this metric has no geometric size penalty: a voxel either is or is not")
    print("  inside the mask. Its chance level does still scale with lesion size, so the lift column")
    print("  is the size-fair comparison, and it is the one that carries the claim.")


if __name__ == "__main__":
    main()
