"""RQ4 evaluation: ensemble across the three size-conditioned text queries, each scored using its
MATCHED physical window size (small->16^3, medium->32^3, large->64^3, all resized to the model's
32^3 input) -- combines RQ2's size-conditioned text with RQ3's multi-scale windowing into one
trained system, per Future Work item 3."""
import csv
import os
import random

import numpy as np
import torch

from dataset import region_mask
from dataset_rq4 import CROP_SIZE_BY_BIN
from evaluate_rq1 import dice_iou, otsu_threshold, size_bin
from localize import sliding_window_heatmap
from model import TextVolumeAligner
from text_encoder import SIZE_CLASS_ORDER

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
TEXT_EMB_PATH = os.path.join(PREPROCESSED_DIR, "size_conditioned_text_embeddings.pt")
CKPT_PATH = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints/rq4_scale_matched_aligner_best.pt"
LESION_CSV = os.path.join(PREPROCESSED_DIR, "lesion_volumes.csv")
RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"

STRIDE_BY_CROP = {16: 8, 32: 16, 64: 32}


def load_true_volumes():
    with open(LESION_CSV, newline="") as f:
        return {row["patient_id"]: row for row in csv.DictReader(f)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    true_volumes = load_true_volumes()

    text_embeds_dict = torch.load(TEXT_EMB_PATH, weights_only=True)
    text_embeds = torch.stack([text_embeds_dict[c] for c in SIZE_CLASS_ORDER])
    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        text_proj = model.encode_text(text_embeds.to(device))
    class_to_idx = {c: i for i, c in enumerate(SIZE_CLASS_ORDER)}

    random.seed(0)
    patient_ids = sorted(f[:-4] for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz"))
    random.shuffle(patient_ids)
    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids = patient_ids[:n_val]
    print(f"evaluating on {len(val_ids)} held-out validation patients")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for region in ["ET", "TC", "WT"]:
        for i, pid in enumerate(val_ids):
            true_vol = float(true_volumes[pid][f"{region.lower()}_volume_mm3"])
            if true_vol <= 0:
                continue

            data = np.load(os.path.join(PREPROCESSED_DIR, f"{pid}.npz"))
            image = torch.from_numpy(data["image"]).float()
            seg = data["mask"]
            gt_mask = region_mask(seg, region)

            # ensemble: each size-phrasing queried at its matched window scale, then take the max
            heatmaps = []
            for bin_label, crop_size in CROP_SIZE_BY_BIN.items():
                idx = class_to_idx[f"{region}_{bin_label}"]
                hm = sliding_window_heatmap(
                    model, image, text_proj[idx], window_size=crop_size, stride=STRIDE_BY_CROP[crop_size],
                    model_input_size=32, device=device,
                ).numpy()
                heatmaps.append(hm)
            heatmap = np.maximum.reduce(heatmaps)

            thresh = otsu_threshold(heatmap.flatten())
            pred_mask = heatmap > thresh
            dice, iou = dice_iou(pred_mask, gt_mask)
            bin_label = size_bin(true_vol, region)

            rows.append({
                "patient_id": pid, "region": region, "size_bin": bin_label,
                "true_volume_mm3": true_vol, "dice": dice, "iou": iou,
            })
            print(f"[{region}] ({i + 1}/{len(val_ids)}) {pid}: bin={bin_label} true_vol={true_vol:.0f}mm3 dice={dice:.3f} iou={iou:.3f}")

    csv_path = os.path.join(RESULTS_DIR, "rq4_localization_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "region", "size_bin", "true_volume_mm3", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-patient scores to {csv_path}")

    print("\n=== RQ4 Summary: mean Dice by region x size bin (scale-matched training + eval) ===")
    for region in ["ET", "TC", "WT"]:
        for b in ["small", "medium", "large"]:
            vals = [r["dice"] for r in rows if r["region"] == region and r["size_bin"] == b]
            if vals:
                print(f"{region} / {b}: n={len(vals)}, mean_dice={np.mean(vals):.3f}")


if __name__ == "__main__":
    main()
