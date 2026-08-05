"""Score the P' supervised segmenter with the project's own size-stratified protocol.

Two questions, one evaluation.

[1] IS THE PIPELINE SOUND? The guideline this project follows is to run the same setup on a
    previously-studied problem and check the numbers land where published work says they should. That
    problem is supervised BraTS segmentation. If a plain U-Net trained on our preprocessed volumes,
    our split, and scored by our dice_iou() reaches Dice in the published BraTS range, then the
    preprocessing, the split, the region definitions and the metric are all working, and the low
    absolute Dice seen in Sections 6-7 is a property of text-conditioned localization rather than a
    bug in the shared machinery.

[2] IS THE SIZE COLLAPSE SPECIFIC TO TEXT CONDITIONING? Every text-conditioned arm degrades on small
    lesions, but no arm has ever been compared against a method that does not use text at all. Scoring
    P' by the identical size_bin() terciles gives a directly comparable large/small Dice ratio. If the
    supervised model also collapses, part of what Sections 6-7 attribute to grounding failure is really
    "small lesions are hard, for everyone"; if it does not, the failure is specific to this task setup.
    Either answer sharpens the paper's central claim, so this is reported whichever way it comes out.

Uses the identical dice_iou(), size_bin() and SIZE_CUTOFFS as every other evaluation script, imported
rather than redefined, so the comparison is not confounded by a differing metric implementation.
"""
import argparse
import csv
import os
import random

import numpy as np
import torch

from dataset_pprime import SEG_REGIONS, BraTSSegDataset
from evaluate_rq1 import PREPROCESSED_DIR, RESULTS_DIR, dice_iou, load_true_volumes, size_bin
from train_pprime_supervised import build_model, get_patient_ids

CKPT_PATH = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints/pprime_unet_best.pt"

# Published BraTS2020 validation-set Dice for reference. These are challenge-leaderboard numbers on
# native-resolution volumes with ensembling and test-time augmentation, so they are an upper anchor,
# not a like-for-like target: our P' is a single small U-Net on 128^3 resampled volumes. The claim
# being checked is "same ballpark", which is what validates the pipeline.
PUBLISHED_REFERENCE = {"ET": (0.70, 0.80), "TC": (0.80, 0.87), "WT": (0.86, 0.89)}


def main():
    """Evaluate the P' checkpoint per region and per size bin, and write the per-patient CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="must match training and RQ1's split")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== P' supervised evaluation | device={device} seed={args.seed} ===")

    true_volumes = load_true_volumes()

    random.seed(args.seed)
    patient_ids = get_patient_ids()
    random.shuffle(patient_ids)
    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids = patient_ids[:n_val]
    print(f"evaluating on {len(val_ids)} held-out patients (identical split to RQ1)")

    model = build_model(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()

    ds = BraTSSegDataset(PREPROCESSED_DIR, val_ids, train=False)
    rows = []
    with torch.no_grad():
        for i, pid in enumerate(val_ids):
            image, target = ds[i]
            probs = torch.sigmoid(model(image.unsqueeze(0).to(device)))[0].cpu().numpy()
            for ri, region in enumerate(SEG_REGIONS):
                true_vol = float(true_volumes[pid][f"{region.lower()}_volume_mm3"])
                if true_vol <= 0:
                    continue  # excluded exactly as in RQ1
                gt = target[ri].numpy() > 0.5
                dice, iou = dice_iou(probs[ri] > args.threshold, gt)
                rows.append({
                    "patient_id": pid, "region": region, "size_bin": size_bin(true_vol, region),
                    "true_volume_mm3": true_vol, "dice": dice, "iou": iou,
                })
            print(f"({i + 1}/{len(val_ids)}) {pid}", flush=True)

    csv_path = os.path.join(RESULTS_DIR, "pprime_supervised_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "region", "size_bin",
                                               "true_volume_mm3", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {csv_path}")

    print("\n[1] PIPELINE VALIDATION: overall Dice vs. published BraTS2020 range")
    print(f"{'region':<8}{'n':>5}{'P-prime Dice':>14}{'published':>16}  in range?")
    for region in SEG_REGIONS:
        vals = [r["dice"] for r in rows if r["region"] == region]
        m = float(np.mean(vals))
        lo, hi = PUBLISHED_REFERENCE[region]
        verdict = "yes" if m >= lo else ("close" if m >= lo - 0.05 else "NO -- investigate")
        published = "{:.2f}-{:.2f}".format(lo, hi)
        print(f"{region:<8}{len(vals):>5}{m:>14.4f}{published:>16}  {verdict}")

    print("\n[2] SIZE STRATIFICATION: does a fully supervised model collapse on small lesions too?")
    print(f"{'region':<8}{'small':>9}{'medium':>9}{'large':>9}{'L/S ratio':>12}")
    for region in SEG_REGIONS:
        means = {}
        for b in ["small", "medium", "large"]:
            vals = [r["dice"] for r in rows if r["region"] == region and r["size_bin"] == b]
            means[b] = float(np.mean(vals)) if vals else float("nan")
        ratio = means["large"] / means["small"] if means["small"] > 0 else float("nan")
        print(f"{region:<8}{means['small']:>9.4f}{means['medium']:>9.4f}{means['large']:>9.4f}"
              f"{ratio:>11.1f}x")
    print("\n    Compare against RQ1's text-conditioned ratios (ET 15.0x, TC 9.2x, WT 4.9x under Otsu;")
    print("    ET 14.4x, TC 5.6x, WT 2.4x under the oracle threshold from Section 6.3).")


if __name__ == "__main__":
    main()
