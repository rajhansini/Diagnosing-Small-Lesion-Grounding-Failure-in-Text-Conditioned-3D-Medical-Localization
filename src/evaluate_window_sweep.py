"""Extends the RQ3b window-size curve: evaluates a single window size in isolation (no ensembling)
on the frozen RQ1 baseline model, for window sizes below the original 32^3, to see where the
"smaller window helps" trend from RQ3b flattens out or reverses."""
import argparse
import csv
import os
import random

import numpy as np
import torch

from dataset import region_mask
from evaluate_rq1 import CKPT_DIR, LESION_CSV, PREPROCESSED_DIR, RESULTS_DIR, TEXT_EMB_PATH, dice_iou, otsu_threshold, size_bin

CKPT_PATH = f"{CKPT_DIR}/baseline_aligner.pt"
from localize import sliding_window_heatmap
from model import TextVolumeAligner
from text_encoder import REGION_ORDER


def main():
    """Evaluate a single isolated window size on the frozen baseline, with no ensembling."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | window_size={args.window_size} stride={args.stride}")

    with open(LESION_CSV, newline="") as f:
        true_volumes = {row["patient_id"]: row for row in csv.DictReader(f)}

    text_embeds_dict = torch.load(TEXT_EMB_PATH, weights_only=True)
    text_embeds = torch.stack([text_embeds_dict[r] for r in REGION_ORDER])
    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        text_proj = model.encode_text(text_embeds.to(device))

    random.seed(0)
    patient_ids = sorted(f[:-4] for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz"))
    random.shuffle(patient_ids)
    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids = patient_ids[:n_val]

    rows = []
    for region in ["ET", "TC", "WT"]:
        region_idx = REGION_ORDER.index(region)
        for pid in val_ids:
            true_vol = float(true_volumes[pid][f"{region.lower()}_volume_mm3"])
            if true_vol <= 0:
                continue
            data = np.load(os.path.join(PREPROCESSED_DIR, f"{pid}.npz"))
            image = torch.from_numpy(data["image"]).float()
            seg = data["mask"]
            gt_mask = region_mask(seg, region)

            heatmap = sliding_window_heatmap(
                model, image, text_proj[region_idx], window_size=args.window_size, stride=args.stride,
                model_input_size=32, device=device,
            ).numpy()
            thresh = otsu_threshold(heatmap.flatten())
            pred_mask = heatmap > thresh
            dice, iou = dice_iou(pred_mask, gt_mask)
            bin_label = size_bin(true_vol, region)
            rows.append({"patient_id": pid, "region": region, "size_bin": bin_label, "dice": dice, "iou": iou})

    print(f"\n=== window={args.window_size} stride={args.stride}: mean Dice by region x size bin ===")
    for region in ["ET", "TC", "WT"]:
        for b in ["small", "medium", "large"]:
            vals = [r["dice"] for r in rows if r["region"] == region and r["size_bin"] == b]
            if vals:
                print(f"{region} / {b}: n={len(vals)}, mean_dice={np.mean(vals):.4f}")

    csv_path = os.path.join(RESULTS_DIR, f"rq3c_window{args.window_size}_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "region", "size_bin", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
