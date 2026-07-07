"""RQ1: size-stratified Dice/IoU evaluation of text-conditioned localization on held-out patients."""
import csv
import os
import random

import numpy as np
import torch

from dataset import region_mask
from localize import sliding_window_heatmap
from model import TextVolumeAligner
from text_encoder import REGION_ORDER

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
TEXT_EMB_PATH = os.path.join(PREPROCESSED_DIR, "subregion_text_embeddings.pt")
CKPT_PATH = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints/baseline_aligner.pt"
LESION_CSV = os.path.join(PREPROCESSED_DIR, "lesion_volumes.csv")
RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"

# tercile cutoffs computed earlier from the full 369-patient lesion_volumes.csv (true mm^3, native resolution)
SIZE_CUTOFFS = {
    "ET": (9050.6, 26245.3),
    "TC": (19190.7, 49948.2),
    "WT": (63126.2, 125309.1),
}


def otsu_threshold(values):
    hist, bin_edges = np.histogram(values, bins=256)
    hist = hist.astype(float)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.clip(weight1, 1e-12, None)
    mean2 = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / np.clip(weight2, 1e-12, None)
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    return bin_centers[idx]


def dice_iou(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    dice = 2 * inter / (pred_mask.sum() + gt_mask.sum() + 1e-8)
    iou = inter / (np.logical_or(pred_mask, gt_mask).sum() + 1e-8)
    return float(dice), float(iou)


def size_bin(volume_mm3, region):
    lo, hi = SIZE_CUTOFFS[region]
    if volume_mm3 <= lo:
        return "small"
    if volume_mm3 <= hi:
        return "medium"
    return "large"


def load_true_volumes():
    with open(LESION_CSV, newline="") as f:
        return {row["patient_id"]: row for row in csv.DictReader(f)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    true_volumes = load_true_volumes()

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
    print(f"evaluating on {len(val_ids)} held-out validation patients")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for region in ["ET", "TC", "WT"]:
        region_idx = REGION_ORDER.index(region)
        for i, pid in enumerate(val_ids):
            true_vol = float(true_volumes[pid][f"{region.lower()}_volume_mm3"])
            if true_vol <= 0:
                continue  # e.g. patients with no ET at all

            data = np.load(os.path.join(PREPROCESSED_DIR, f"{pid}.npz"))
            image = torch.from_numpy(data["image"]).float()
            seg = data["mask"]
            gt_mask = region_mask(seg, region)

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

    csv_path = os.path.join(RESULTS_DIR, "rq1_localization_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "region", "size_bin", "true_volume_mm3", "dice", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-patient scores to {csv_path}")

    print("\n=== Summary: mean Dice by region x size bin ===")
    for region in ["ET", "TC", "WT"]:
        for bin_label in ["small", "medium", "large"]:
            vals = [r["dice"] for r in rows if r["region"] == region and r["size_bin"] == bin_label]
            if vals:
                print(f"{region} / {bin_label}: n={len(vals)}, mean_dice={np.mean(vals):.3f}")


if __name__ == "__main__":
    main()
