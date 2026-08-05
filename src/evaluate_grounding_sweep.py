"""RQ12: does the best intervention fix the *grounding* failure, or only the boundary?

Two loose ends from RQ11 are closed here at once, because both need the same recomputed heatmaps.

(1) THE POINTING QUESTION. RQ11's sharpest result was threshold-free: at ET-small the frozen 32^3
    baseline's peak response landed inside the true lesion in 0 of 21 patients, a median 27.5mm away
    -- exactly chance. Every ablation in Section 7, including the smaller-window fix that won overall,
    was scored only by Dice. Dice conflates "pointed at the lesion" with "drew the right boundary", so
    RQ3c's win is currently consistent with two very different stories: it genuinely found lesions it
    used to miss, or it merely tightened masks around lesions it was already finding. The pointing
    game separates them, and it has no geometric size penalty.

(2) THE THRESHOLD QUESTION. Section 7's ablation comparisons all share the Otsu binarizer, and
    Section 10 argues they are therefore internally fair. That is an assumption, not a measurement:
    RQ11 showed Otsu is *anti*-correlated with lesion size (rho = -0.26 to -0.48), so it is not a
    neutral common factor. Re-scoring the winning intervention under the deployable fixed-percentile
    rule tests whether "smaller window wins" survives a better-calibrated threshold.

This script generalizes evaluate_rq11_threshold_confound.py to an arbitrary query window size, so
running it at --window_size 32 must reproduce results/rq11_threshold_confound_scores.csv exactly.
That reproduction is an end-to-end correctness gate on the generalization and is checked automatically
by analyze_rq12.py, not asserted by hand.

Geometry note (identical to RQ11): the preprocessed grid is 128^3 resampled from a native
240x240x155 volume at 1mm^3 isotropic, so a preprocessed voxel spans (1.875, 1.875, 1.2109) mm and
occupies 4.2572 mm^3. Peak-to-lesion distances are anisotropic and converted per-axis.
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from dataset import region_mask
from evaluate_rq1 import (
    CKPT_DIR,
    PREPROCESSED_DIR,
    RESULTS_DIR,
    TEXT_EMB_PATH,
    dice_iou,
    load_true_volumes,
    otsu_threshold,
    size_bin,
)
from evaluate_rq11_threshold_confound import PERCENTILES, SPACING, VOXEL_VOLUME_MM3, top_k_mask
from localize import sliding_window_heatmap
from model import TextVolumeAligner
from text_encoder import REGION_ORDER

CKPT_PATH = os.path.join(CKPT_DIR, "baseline_aligner.pt")


def main():
    """Recompute heatmaps at one window size and score them under five binarizers plus the pointing game."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, required=True,
                        help="physical query window swept over the volume; 32 reproduces RQ11")
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--limit_patients", type=int, default=None,
                        help="smoke tests only: truncate the val set AFTER the split is computed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RQ12 grounding sweep | device={device} window={args.window_size} stride={args.stride} ===")

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
    if args.limit_patients:
        val_ids = val_ids[: args.limit_patients]
        print(f"SMOKE TEST: truncated to {len(val_ids)} validation patients")
    print(f"evaluating on {len(val_ids)} held-out patients (seed 0 split, identical to RQ1/RQ11)")

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
            gt_voxels = int(gt_mask.sum())
            bin_label = size_bin(true_vol, region)

            heatmap = sliding_window_heatmap(
                model, image, text_proj[region_idx],
                window_size=args.window_size, stride=args.stride,
                model_input_size=32, device=device,
            ).numpy()

            # Tie-aware peak location. When stride == window_size the windows do not overlap, so every
            # voxel is covered exactly once and the heatmap is piecewise-constant over window-sized
            # blocks. np.argmax then returns the *first* voxel of the winning block in C-order -- its
            # corner -- rather than anything meaningfully "peak-like", which biases the reported peak
            # several voxels away from the block's centre and can push it outside a small lesion even
            # when the block is centred on one. RQ11's overlapping 32/16 protocol never hit this, but
            # comparing window sizes does, so we record the plateau explicitly instead of hiding it.
            dist_field = distance_transform_edt(~gt_mask, sampling=SPACING) if gt_voxels > 0 else None
            tied_flat = np.flatnonzero(heatmap.ravel() == heatmap.max())
            tied = np.array(np.unravel_index(tied_flat, heatmap.shape)).T  # (n_tied, 3)
            n_tied = int(tied.shape[0])

            peak_idx = tuple(int(v) for v in tied[0])          # RQ11-compatible first-argmax
            argmax_hit = bool(gt_mask[peak_idx])
            dist_mm = float(dist_field[peak_idx]) if dist_field is not None else float("nan")

            centroid = tuple(int(round(float(v))) for v in tied.mean(axis=0))
            centroid_hit = bool(gt_mask[centroid])
            centroid_dist_mm = float(dist_field[centroid]) if dist_field is not None else float("nan")

            any_tied_hit = bool(gt_mask[tuple(tied.T)].any())

            candidates = {"otsu": heatmap > otsu_threshold(heatmap.flatten())}
            for pct in PERCENTILES:
                candidates[f"pct{int(pct)}"] = top_k_mask(heatmap, round(heatmap.size * (100.0 - pct) / 100.0))
            candidates["oracle_volume"] = top_k_mask(heatmap, gt_voxels)

            for method, pred_mask in candidates.items():
                dice, iou = dice_iou(pred_mask, gt_mask)
                pred_voxels = int(pred_mask.sum())
                rows.append({
                    "patient_id": pid, "region": region, "size_bin": bin_label,
                    "window_size": args.window_size,
                    "threshold_method": method,
                    "true_volume_mm3": true_vol,
                    "gt_voxels_resized": gt_voxels,
                    "gt_volume_mm3_resized": gt_voxels * VOXEL_VOLUME_MM3,
                    "pred_voxels": pred_voxels,
                    "pred_volume_mm3": pred_voxels * VOXEL_VOLUME_MM3,
                    "dice": dice, "iou": iou,
                    "argmax_hit": int(argmax_hit),
                    "argmax_dist_mm": dist_mm,
                    "peak_tie_count": n_tied,
                    "centroid_hit": int(centroid_hit),
                    "centroid_dist_mm": centroid_dist_mm,
                    "any_tied_hit": int(any_tied_hit),
                })

            otsu_row = next(r for r in rows[-len(candidates):] if r["threshold_method"] == "otsu")
            pct99_row = next(r for r in rows[-len(candidates):] if r["threshold_method"] == "pct99")
            print(f"[{region}] ({i + 1}/{len(val_ids)}) {pid}: bin={bin_label} "
                  f"dice_otsu={otsu_row['dice']:.3f} dice_pct99={pct99_row['dice']:.3f} "
                  f"peak_hit={int(argmax_hit)} peak_dist={dist_mm:.1f}mm "
                  f"ties={n_tied} centroid_hit={int(centroid_hit)}", flush=True)

    fieldnames = ["patient_id", "region", "size_bin", "window_size", "threshold_method",
                  "true_volume_mm3", "gt_voxels_resized", "gt_volume_mm3_resized",
                  "pred_voxels", "pred_volume_mm3", "dice", "iou", "argmax_hit", "argmax_dist_mm",
                  "peak_tie_count", "centroid_hit", "centroid_dist_mm", "any_tied_hit"]
    # Stride is part of the filename, not just the window size: RQ3c switched from 50%-overlap to
    # non-overlapping tiling partway down the sweep, so (window, stride) pairs are distinct
    # conditions and a window-only filename would silently overwrite one with the other.
    suffix = "_smoke" if args.limit_patients else ""
    csv_path = os.path.join(
        RESULTS_DIR,
        f"rq12_grounding_window{args.window_size}_stride{args.stride}{suffix}_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {csv_path}")

    print(f"\n=== window={args.window_size}: pointing game by region x size bin ===")
    print("argmax = RQ11's first-argmax rule; centroid = centre of the tied-maximum plateau.")
    print("Where the two disagree, the plateau is large enough that 'the peak voxel' is ambiguous.")
    print(f"{'region':<7}{'bin':<8}{'n':>4}{'argmax':>9}{'centroid':>10}{'any_tied':>10}{'med_ties':>10}")
    for region in ["ET", "TC", "WT"]:
        for b in ["small", "medium", "large"]:
            sel = [r for r in rows if r["region"] == region and r["size_bin"] == b
                   and r["threshold_method"] == "otsu"]
            if sel:
                n = len(sel)
                print(f"{region:<7}{b:<8}{n:>4}"
                      f"{sum(r['argmax_hit'] for r in sel) / n:>9.3f}"
                      f"{sum(r['centroid_hit'] for r in sel) / n:>10.3f}"
                      f"{sum(r['any_tied_hit'] for r in sel) / n:>10.3f}"
                      f"{int(np.median([r['peak_tie_count'] for r in sel])):>10d}")


if __name__ == "__main__":
    main()
