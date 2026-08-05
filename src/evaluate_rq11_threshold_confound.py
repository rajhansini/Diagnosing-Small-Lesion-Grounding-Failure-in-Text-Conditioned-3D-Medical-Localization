"""RQ11: is the size-dependent Dice collapse a *grounding* failure or a *thresholding* artifact?

The sharpest available attack on this project's headline result is that Otsu binarizes every volume
independently, so the predicted mask's size is set by the heatmap's intensity histogram and not by
the text query at all. If Otsu emits a roughly constant-size blob regardless of the true lesion
volume, then small lesions would score badly for a reason that has nothing to do with text-conditioned
grounding, and the RQ1 collapse would be partly an artifact of the binarization step. The existing
result CSVs cannot answer this: they never record how large the predicted mask actually was.

This script computes the heatmap ONCE per (patient, region) with the frozen RQ1 baseline and the
identical 32^3 / stride-16 protocol, then binarizes it five different ways:

  otsu           the existing protocol, reproduced exactly (reference condition)
  pct90/95/99    fixed top-k% of voxels -- prediction size is constant BY CONSTRUCTION, so any
                 residual size collapse here cannot be attributed to adaptive thresholding
  oracle_volume  top-k where k is exactly the ground-truth voxel count. This removes threshold
                 calibration from the problem entirely and asks only: how good is the heatmap's
                 *ranking* of voxels? It is an oracle (uses the true volume) and therefore an upper
                 bound / diagnostic, never a deployable method -- reported as such.

If the collapse persists under oracle_volume, the heatmap genuinely ranks small-lesion voxels badly
and the grounding failure is real. If it vanishes, the finding was substantially a thresholding
artifact and the paper's central claim needs restating. Either way we should know before a reviewer
asks.

Also records, free of charge from the same heatmap, two threshold-independent grounding metrics:
  argmax_hit           is the single highest-response voxel inside the true mask? (the "pointing
                       game" used in weakly-supervised grounding -- unlike Dice it has no built-in
                       size bias, so it separates "points at the lesion" from "draws its boundary")
  argmax_dist_mm       physical distance from that peak voxel to the nearest ground-truth voxel

Geometry note: the preprocessed grid is 128^3 resampled from a native 240x240x155 volume at 1mm^3
isotropic, so a preprocessed voxel spans (1.875, 1.875, 1.2109) mm and occupies 4.2572 mm^3. Voxel
distances are anisotropic and are converted per-axis, not by a single scalar.
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
from scipy import stats
from scipy.ndimage import distance_transform_edt

from dataset import region_mask
from evaluate_rq1 import dice_iou, load_true_volumes, otsu_threshold, size_bin
from localize import sliding_window_heatmap
from model import TextVolumeAligner
from text_encoder import REGION_ORDER

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
TEXT_EMB_PATH = os.path.join(PREPROCESSED_DIR, "subregion_text_embeddings.pt")
CKPT_PATH = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints/baseline_aligner.pt"
RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"

NATIVE_SHAPE = (240, 240, 155)  # BraTS2020 native grid, 1mm^3 isotropic
GRID = 128
SPACING = tuple(n / GRID for n in NATIVE_SHAPE)  # mm per preprocessed voxel, per axis
VOXEL_VOLUME_MM3 = float(np.prod(SPACING))  # 4.2572 mm^3

PERCENTILES = [90.0, 95.0, 99.0]


def top_k_mask(heatmap, k):
    """Exactly k highest-valued voxels. Uses argpartition rather than a value threshold so ties at
    the cut point cannot make the mask larger or smaller than k."""
    flat = heatmap.ravel()
    k = int(min(max(k, 0), flat.size))
    mask = np.zeros(flat.size, dtype=bool)
    if k > 0:
        mask[np.argpartition(flat, -k)[-k:]] = True
    return mask.reshape(heatmap.shape)


def main():
    """Recompute each heatmap once and score it under five binarizers plus the pointing game."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_patients", type=int, default=None,
                        help="smoke tests only: truncate the val set AFTER the split is computed. "
                             "Results are written to a _smoke CSV.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RQ11 threshold-confound analysis | device={device} ===")
    print(f"voxel spacing (mm): {tuple(round(s, 4) for s in SPACING)} | voxel volume: {VOXEL_VOLUME_MM3:.4f} mm^3")

    true_volumes = load_true_volumes()

    text_embeds_dict = torch.load(TEXT_EMB_PATH, weights_only=True)
    text_embeds = torch.stack([text_embeds_dict[r] for r in REGION_ORDER])
    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        text_proj = model.encode_text(text_embeds.to(device))
    print(f"loaded frozen RQ1 baseline from {CKPT_PATH}")

    random.seed(0)
    patient_ids = sorted(f[:-4] for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz"))
    random.shuffle(patient_ids)
    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids = patient_ids[:n_val]
    if args.limit_patients:
        val_ids = val_ids[: args.limit_patients]
        print(f"SMOKE TEST: truncated to {len(val_ids)} validation patients")
    print(f"evaluating on {len(val_ids)} held-out validation patients (seed 0 split, identical to RQ1)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    n_empty_gt = 0
    for region in ["ET", "TC", "WT"]:
        region_idx = REGION_ORDER.index(region)
        for i, pid in enumerate(val_ids):
            true_vol = float(true_volumes[pid][f"{region.lower()}_volume_mm3"])
            if true_vol <= 0:
                continue

            data = np.load(os.path.join(PREPROCESSED_DIR, f"{pid}.npz"))
            image = torch.from_numpy(data["image"]).float()
            gt_mask = region_mask(data["mask"], region)
            gt_voxels = int(gt_mask.sum())
            bin_label = size_bin(true_vol, region)

            # A lesion can be non-empty at native resolution yet vanish on the 128^3 grid under
            # nearest-neighbour mask resizing. Dice is then 0 no matter what the model does, which
            # is itself worth surfacing rather than silently averaging in.
            if gt_voxels == 0:
                n_empty_gt += 1
                print(f"[{region}] {pid}: WARNING gt mask empty on 128^3 grid "
                      f"(native volume {true_vol:.0f}mm^3, bin={bin_label}) -- Dice is 0 by construction")

            heatmap = sliding_window_heatmap(model, image, text_proj[region_idx], patch_size=32, stride=16, device=device).numpy()

            # Threshold-independent grounding: where does the single peak response land?
            peak_idx = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
            argmax_hit = bool(gt_mask[peak_idx])
            if gt_voxels > 0:
                dist_mm = float(distance_transform_edt(~gt_mask, sampling=SPACING)[peak_idx])
            else:
                dist_mm = float("nan")

            candidates = {"otsu": heatmap > otsu_threshold(heatmap.flatten())}
            for pct in PERCENTILES:
                candidates[f"pct{int(pct)}"] = top_k_mask(heatmap, round(heatmap.size * (100.0 - pct) / 100.0))
            candidates["oracle_volume"] = top_k_mask(heatmap, gt_voxels)

            for method, pred_mask in candidates.items():
                dice, iou = dice_iou(pred_mask, gt_mask)
                pred_voxels = int(pred_mask.sum())
                rows.append({
                    "patient_id": pid, "region": region, "size_bin": bin_label,
                    "threshold_method": method,
                    "true_volume_mm3": true_vol,
                    "gt_voxels_resized": gt_voxels,
                    "gt_volume_mm3_resized": gt_voxels * VOXEL_VOLUME_MM3,
                    "pred_voxels": pred_voxels,
                    "pred_volume_mm3": pred_voxels * VOXEL_VOLUME_MM3,
                    "dice": dice, "iou": iou,
                    "argmax_hit": int(argmax_hit),
                    "argmax_dist_mm": dist_mm,
                })

            otsu_row = next(r for r in rows[-len(candidates):] if r["threshold_method"] == "otsu")
            oracle_row = next(r for r in rows[-len(candidates):] if r["threshold_method"] == "oracle_volume")
            print(f"[{region}] ({i + 1}/{len(val_ids)}) {pid}: bin={bin_label} "
                  f"gt={gt_voxels}vox otsu_pred={otsu_row['pred_voxels']}vox "
                  f"dice_otsu={otsu_row['dice']:.3f} dice_oracle={oracle_row['dice']:.3f} "
                  f"peak_hit={int(argmax_hit)} peak_dist={dist_mm:.1f}mm")

    fieldnames = ["patient_id", "region", "size_bin", "threshold_method", "true_volume_mm3",
                  "gt_voxels_resized", "gt_volume_mm3_resized", "pred_voxels", "pred_volume_mm3",
                  "dice", "iou", "argmax_hit", "argmax_dist_mm"]
    csv_path = os.path.join(RESULTS_DIR, "rq11_threshold_confound_scores%s.csv" % ("_smoke" if args.limit_patients else ""))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {csv_path}")
    if n_empty_gt:
        print(f"NOTE: {n_empty_gt} (patient, region) pairs had an empty ground-truth mask after resizing.")

    # ---- Summary 1: does Otsu's predicted size actually track the lesion's size? ----
    print("\n=== Does the predicted mask size track the true lesion size? ===")
    print("(If Otsu's predicted volume is ~constant across size bins, Dice collapse is partly a")
    print(" thresholding artifact rather than a grounding failure.)")
    for method in ["otsu", "oracle_volume"]:
        print(f"\n-- {method} --")
        for region in ["ET", "TC", "WT"]:
            sel = [r for r in rows if r["region"] == region and r["threshold_method"] == method]
            if len(sel) < 5:
                continue
            true_v = [r["true_volume_mm3"] for r in sel]
            pred_v = [r["pred_volume_mm3"] for r in sel]
            rho, p = stats.spearmanr(true_v, pred_v)
            line = f"  {region}: spearman(true_vol, pred_vol) rho={rho:+.3f} (p={p:.2e}) | mean pred volume by bin:"
            for b in ["small", "medium", "large"]:
                vals = [r["pred_volume_mm3"] for r in sel if r["size_bin"] == b]
                if vals:
                    line += f" {b}={np.mean(vals):,.0f}mm^3"
            print(line)

    # ---- Summary 2: does the size collapse survive each binarization rule? ----
    print("\n=== Mean Dice by region x size bin, under every threshold rule ===")
    header = f"{'region':<5} {'method':<14}" + "".join(f"{b:>10}" for b in ["small", "medium", "large"]) + f"{'L/S ratio':>11}"
    print(header)
    for region in ["ET", "TC", "WT"]:
        for method in ["otsu"] + [f"pct{int(p)}" for p in PERCENTILES] + ["oracle_volume"]:
            means = {}
            for b in ["small", "medium", "large"]:
                vals = [r["dice"] for r in rows if r["region"] == region and r["size_bin"] == b and r["threshold_method"] == method]
                means[b] = np.mean(vals) if vals else float("nan")
            ratio = means["large"] / means["small"] if means["small"] > 0 else float("nan")
            print(f"{region:<5} {method:<14}" + "".join(f"{means[b]:>10.4f}" for b in ["small", "medium", "large"]) + f"{ratio:>11.1f}x")

    # ---- Summary 3: the threshold-independent grounding metrics ----
    print("\n=== Threshold-independent grounding: pointing game (peak-response voxel) ===")
    print(f"{'region':<5} {'bin':<8} {'n':>4} {'hit rate':>10} {'mean dist (mm)':>16} {'median dist':>13}")
    for region in ["ET", "TC", "WT"]:
        for b in ["small", "medium", "large"]:
            sel = [r for r in rows if r["region"] == region and r["size_bin"] == b and r["threshold_method"] == "otsu"]
            if not sel:
                continue
            hits = [r["argmax_hit"] for r in sel]
            dists = [r["argmax_dist_mm"] for r in sel if not np.isnan(r["argmax_dist_mm"])]
            mean_d = np.mean(dists) if dists else float("nan")
            med_d = np.median(dists) if dists else float("nan")
            print(f"{region:<5} {b:<8} {len(sel):>4} {np.mean(hits):>10.3f} {mean_d:>16.1f} {med_d:>13.1f}")
    print("\nA flat hit rate across size bins would mean the model points at small lesions just as")
    print("reliably as large ones, and that the collapse is specific to boundary delineation.")


if __name__ == "__main__":
    main()
