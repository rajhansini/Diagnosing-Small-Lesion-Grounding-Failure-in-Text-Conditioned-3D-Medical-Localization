"""RQ7 evaluation: size-stratified Dice/IoU for a text-encoder-ablated model.

Byte-for-byte the same localization protocol as evaluate_rq1.py -- same frozen sliding window
(32^3, stride 16), same Otsu binarization, same seed-0 held-out patients, same size-bin cutoffs,
same reported columns -- so the output CSV is directly paired against
results/rq1_localization_scores.csv patient-by-patient for Wilcoxon testing.

Reuses otsu_threshold / dice_iou / size_bin imported from evaluate_rq1 rather than reimplementing
them, so there is no possibility of the two evaluation paths silently diverging.
"""
import argparse
import csv
import os
import random

import numpy as np
import torch

from dataset import region_mask
from evaluate_rq1 import dice_iou, load_true_volumes, otsu_threshold, size_bin
from localize import sliding_window_heatmap
from model import TextVolumeAligner
from text_encoder import REGION_ORDER

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
CKPT_DIR = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints"
RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
# "whitened" is RQ7's missing cell: PubMedBERT's real embeddings with the anisotropy
# removed by an invertible whitening map, so it keeps the semantics and discards only
# the geometry -- the one combination the original four conditions never isolated.
VARIANTS = ["bertbase", "randbert", "randvec_ortho", "randvec_aniso", "whitened"]


def main():
    """Evaluate one text-encoder ablation under the identical RQ1 localization protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--ckpt", default="last", choices=["last", "best"],
                        help="'last' is the pre-registered primary comparison: baseline_aligner.pt is "
                             "itself a last-epoch checkpoint, so last-vs-last holds checkpoint "
                             "selection fixed across conditions. 'best' is a robustness appendix.")
    parser.add_argument("--seed", type=int, default=0, help="must match the training split seed")
    parser.add_argument("--limit_patients", type=int, default=None,
                        help="smoke tests only: truncate the val set AFTER the split is computed, so the "
                             "split itself is unchanged. Results are written to a _smoke CSV.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RQ7 eval | variant={args.variant} | ckpt={args.ckpt} | device={device} | seed={args.seed} ===")

    true_volumes = load_true_volumes()

    text_emb_path = os.path.join(PREPROCESSED_DIR, f"rq7_{args.variant}_text_embeddings.pt")
    text_embeds_dict = torch.load(text_emb_path, weights_only=True)
    text_embeds = torch.stack([text_embeds_dict[r] for r in REGION_ORDER])

    stem = f"rq7_{args.variant}_aligner" if args.seed == 0 else f"rq7_{args.variant}_seed{args.seed}_aligner"
    ckpt_path = os.path.join(CKPT_DIR, f"{stem}_{args.ckpt}.pt")
    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        text_proj = model.encode_text(text_embeds.to(device))
    print(f"loaded {ckpt_path}")

    random.seed(args.seed)
    patient_ids = sorted(f[:-4] for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz"))
    random.shuffle(patient_ids)
    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids = patient_ids[:n_val]
    if args.limit_patients:
        val_ids = val_ids[: args.limit_patients]
        print(f"SMOKE TEST: truncated to {len(val_ids)} validation patients")
    print(f"evaluating on {len(val_ids)} held-out validation patients")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for region in ["ET", "TC", "WT"]:
        region_idx = REGION_ORDER.index(region)
        for i, pid in enumerate(val_ids):
            true_vol = float(true_volumes[pid][f"{region.lower()}_volume_mm3"])
            if true_vol <= 0:
                continue

            data = np.load(os.path.join(PREPROCESSED_DIR, f"{pid}.npz"))
            image = torch.from_numpy(data["image"]).float()
            gt_mask = region_mask(data["mask"], region)

            heatmap = sliding_window_heatmap(model, image, text_proj[region_idx], patch_size=32, stride=16, device=device).numpy()
            thresh = otsu_threshold(heatmap.flatten())
            pred_mask = heatmap > thresh
            dice, iou = dice_iou(pred_mask, gt_mask)
            bin_label = size_bin(true_vol, region)

            rows.append({
                "patient_id": pid, "region": region, "size_bin": bin_label,
                "true_volume_mm3": true_vol, "dice": dice, "iou": iou,
            })
            print(f"[{region}] ({i + 1}/{len(val_ids)}) {pid}: bin={bin_label} true_vol={true_vol:.0f}mm3 dice={dice:.3f} iou={iou:.3f}")

    suffix = "" if args.ckpt == "last" else f"_{args.ckpt}"
    if args.seed != 0:
        suffix += f"_seed{args.seed}"
    if args.limit_patients:
        suffix += "_smoke"
    csv_path = os.path.join(RESULTS_DIR, f"rq7_{args.variant}{suffix}_localization_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "region", "size_bin", "true_volume_mm3", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-patient scores to {csv_path}")

    print(f"\n=== RQ7 ({args.variant}, {args.ckpt}) Summary: mean Dice by region x size bin ===")
    for region in ["ET", "TC", "WT"]:
        for b in ["small", "medium", "large"]:
            vals = [r["dice"] for r in rows if r["region"] == region and r["size_bin"] == b]
            if vals:
                print(f"{region} / {b}: n={len(vals)}, mean_dice={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
