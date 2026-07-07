"""RQ3b: isolate whether a smaller window ALONE (16^3, no ensembling with other scales) helps,
separating "is a smaller receptive field better" from "does naively ensembling across scales hurt"."""
import evaluate_rq3_multiscale as m

if __name__ == "__main__":
    m.SCALES = [(16, 8)]
    import csv
    import os

    import numpy as np
    import torch

    from dataset import region_mask
    from evaluate_rq1 import dice_iou, otsu_threshold, size_bin
    from localize import sliding_window_heatmap
    from model import TextVolumeAligner
    from text_encoder import REGION_ORDER
    import random

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| scale: 16-only")

    with open(m.LESION_CSV, newline="") as f:
        true_volumes = {row["patient_id"]: row for row in csv.DictReader(f)}

    text_embeds_dict = torch.load(m.TEXT_EMB_PATH, weights_only=True)
    text_embeds = torch.stack([text_embeds_dict[r] for r in REGION_ORDER])
    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(device)
    model.load_state_dict(torch.load(m.CKPT_PATH, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        text_proj = model.encode_text(text_embeds.to(device))

    random.seed(0)
    patient_ids = sorted(f[:-4] for f in os.listdir(m.PREPROCESSED_DIR) if f.endswith(".npz"))
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
            data = np.load(os.path.join(m.PREPROCESSED_DIR, f"{pid}.npz"))
            image = torch.from_numpy(data["image"]).float()
            seg = data["mask"]
            gt_mask = region_mask(seg, region)

            heatmap = sliding_window_heatmap(
                model, image, text_proj[region_idx], window_size=16, stride=8, model_input_size=32, device=device,
            ).numpy()
            thresh = otsu_threshold(heatmap.flatten())
            pred_mask = heatmap > thresh
            dice, iou = dice_iou(pred_mask, gt_mask)
            bin_label = size_bin(true_vol, region)
            rows.append({"patient_id": pid, "region": region, "size_bin": bin_label, "dice": dice})

    print("\n=== 16-only (no ensemble): mean Dice by region x size bin ===")
    for region in ["ET", "TC", "WT"]:
        for b in ["small", "medium", "large"]:
            vals = [r["dice"] for r in rows if r["region"] == region and r["size_bin"] == b]
            if vals:
                print(f"{region} / {b}: n={len(vals)}, mean_dice={np.mean(vals):.4f}")

    csv_path = "/net/projects/ranalab/rajhansini/nlp_project/results/rq3b_scale16only_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "region", "size_bin", "dice"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {csv_path}")
