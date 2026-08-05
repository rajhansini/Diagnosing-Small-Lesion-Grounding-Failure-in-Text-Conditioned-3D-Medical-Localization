"""Chance baseline: random heatmaps run through the identical Otsu+Dice pipeline, on the exact
same (patient, region) pairs evaluated in RQ1. Tests whether the observed small-lesion Dice
collapse is a real model failure or simply an artifact of Dice's known sensitivity to small
structures (any method, even pure noise, tends to score lower on small targets)."""
import csv
import os

import numpy as np

from dataset import region_mask
from evaluate_rq1 import dice_iou, otsu_threshold

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"


def main():
    """Score random heatmaps through the identical Otsu+Dice pipeline as a chance-level control.

    Separates genuine model failure from Dice's known geometric bias against small structures: without
    this, "small lesions score worse" is a property of the metric, not evidence about the model.
    """
    rng = np.random.default_rng(0)

    with open(os.path.join(RESULTS_DIR, "rq1_localization_scores.csv"), newline="") as f:
        rq1_rows = list(csv.DictReader(f))

    rows = []
    for r in rq1_rows:
        pid, region = r["patient_id"], r["region"]
        seg = np.load(os.path.join(PREPROCESSED_DIR, f"{pid}.npz"))["mask"]
        gt_mask = region_mask(seg, region)

        heatmap = rng.normal(size=seg.shape)  # pure noise, no model, no text
        thresh = otsu_threshold(heatmap.flatten())
        pred_mask = heatmap > thresh
        dice, iou = dice_iou(pred_mask, gt_mask)

        rows.append({
            "patient_id": pid, "region": region, "size_bin": r["size_bin"],
            "true_volume_mm3": r["true_volume_mm3"], "dice": dice, "iou": iou,
        })

    csv_path = os.path.join(RESULTS_DIR, "chance_baseline_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "region", "size_bin", "true_volume_mm3", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)

    print("=== Chance baseline (random heatmap): mean Dice by region x size bin ===")
    for region in ["ET", "TC", "WT"]:
        for b in ["small", "medium", "large"]:
            vals = [x["dice"] for x in rows if x["region"] == region and x["size_bin"] == b]
            if vals:
                print(f"{region} / {b}: n={len(vals)}, mean_dice={np.mean(vals):.4f}")
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
