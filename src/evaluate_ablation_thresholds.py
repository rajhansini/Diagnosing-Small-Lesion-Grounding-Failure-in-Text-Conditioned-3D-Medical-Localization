"""RQ13: re-score the retrained ablations (RQ2/RQ4/RQ6) under all five binarization rules.

Section 7.11 established that Otsu is not a neutral common factor: at a fixed window size it and every
calibrated rule disagree in *sign* about which of two heatmaps is better. That finding was obtained on
the inference-side arms (the window sweep). The retrained arms -- RQ2's size-conditioned text ensemble,
RQ4's scale-matched ensemble and RQ6's uniformity-regularized version of it -- have still only ever
been scored under Otsu, so their verdicts in Sections 7.1, 7.5 and 7.6 carry exactly the same risk:
each builds its heatmap by a voxel-wise MAX over several queries, which changes the intensity
distribution Otsu keys on, and different arms take that max over different numbers of maps.

This script reproduces each arm's published heatmap construction exactly, then scores the resulting
heatmap under otsu / pct90 / pct95 / pct99 / oracle_volume plus the tie-aware pointing game, so the
Section 7 rankings can be checked against a calibrated threshold rather than assumed to be fair.

The Otsu column must reproduce that arm's existing `rq{N}_localization_scores.csv` to floating point;
that is an end-to-end correctness gate on this script and is checked by analyze_rq13.py.
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from dataset import region_mask
from dataset_rq4 import CROP_SIZE_BY_BIN
from evaluate_rq1 import (
    CKPT_DIR,
    PREPROCESSED_DIR,
    RESULTS_DIR,
    dice_iou,
    load_true_volumes,
    otsu_threshold,
    size_bin,
)
from evaluate_rq11_threshold_confound import PERCENTILES, SPACING, VOXEL_VOLUME_MM3, top_k_mask
from localize import sliding_window_heatmap
from model import TextVolumeAligner
from text_encoder import SIZE_CLASS_ORDER

# Matches evaluate_rq4.py / evaluate_rq6.py: each matched crop is swept at 50% overlap.
STRIDE_BY_CROP = {16: 8, 32: 16, 64: 32}

PREPROC = PREPROCESSED_DIR
SIZE_EMB_PATH = os.path.join(PREPROC, "size_conditioned_text_embeddings.pt")

# arm -> (checkpoint basename, seeded-checkpoint pattern, scale-matched?)
ARMS = {
    "rq2": ("rq2_size_conditioned_aligner_best.pt", "rq2_size_conditioned_seed{s}_aligner_best.pt", False),
    "rq4": ("rq4_scale_matched_aligner_best.pt", "rq4_scale_matched_seed{s}_aligner_best.pt", True),
    "rq6": ("rq6_uniformity_aligner_best.pt", "rq6_uniformity_seed{s}_aligner_best.pt", True),
}


def build_heatmap(model, image, text_proj, class_to_idx, region, scale_matched, device,
                  weighting="uniform"):
    """Reproduce one arm's published heatmap: a voxel-wise max over the size-phrasing ensemble.

    RQ2 queries all three size phrasings at the model's native 32³ window. RQ4 and RQ6 additionally
    query each phrasing at its matched physical window (small->16³, medium->32³, large->64³, each at
    50% overlap), which is the protocol Sections 7.5-7.6 used.

    Args:
        model: the trained TextVolumeAligner for this arm.
        image: (4, D, H, W) preprocessed volume.
        text_proj: (10, proj_dim) projected size-conditioned class embeddings.
        class_to_idx: mapping from "REGION_bin" to its row in text_proj.
        region: ET/TC/WT.
        scale_matched: True for RQ4/RQ6, False for RQ2.
        device: torch device string.
        weighting: accumulation rule passed to sliding_window_heatmap. "uniform" reproduces the
            published Section 7 protocol; "gaussian" applies the centre-weighted read-out RQ15
            showed is worth +0.145 Dice at the baseline. Since these arms build their heatmap by a
            voxel-wise maximum over three queries, and the read-out changes each query's spatial
            profile before that maximum is taken, the arms need not respond to it the way the
            single-query baseline does -- which is the whole reason for re-scoring them.

    Returns:
        (D, H, W) float ndarray, the ensemble heatmap.
    """
    maps = []
    for bin_label in ["small", "medium", "large"]:
        idx = class_to_idx[f"{region}_{bin_label}"]
        if scale_matched:
            crop = CROP_SIZE_BY_BIN[bin_label]
            hm = sliding_window_heatmap(model, image, text_proj[idx], window_size=crop,
                                        stride=STRIDE_BY_CROP[crop], model_input_size=32,
                                        device=device, weighting=weighting)
        else:
            hm = sliding_window_heatmap(model, image, text_proj[idx], patch_size=32, stride=16,
                                        device=device, weighting=weighting)
        maps.append(hm.numpy())
    return np.maximum.reduce(maps)


def main():
    """Score one retrained ablation under all five binarizers plus the tie-aware pointing game."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq", required=True, choices=sorted(ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weighting", choices=["uniform", "gaussian"], default="uniform",
                        help="accumulation rule; uniform reproduces the published Section 7 scores")
    parser.add_argument("--max_hours", type=float, default=None,
                        help="stop cleanly before the SLURM wall and exit 42 so the job script can "
                             "resubmit; completed (patient, region) pairs are already on disk")
    parser.add_argument("--limit_patients", type=int, default=None, help="smoke tests only")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_name, seed_pattern, scale_matched = ARMS[args.rq]
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name if args.seed == 0
                             else seed_pattern.format(s=args.seed))
    print(f"=== RQ13 threshold re-score | arm={args.rq} seed={args.seed} device={device} ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"scale-matched ensemble: {scale_matched}")

    true_volumes = load_true_volumes()
    text_embeds_dict = torch.load(SIZE_EMB_PATH, weights_only=True)
    text_embeds = torch.stack([text_embeds_dict[c] for c in SIZE_CLASS_ORDER])

    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        text_proj = model.encode_text(text_embeds.to(device))
    class_to_idx = {c: i for i, c in enumerate(SIZE_CLASS_ORDER)}

    random.seed(args.seed)
    patient_ids = sorted(f[:-4] for f in os.listdir(PREPROC) if f.endswith(".npz"))
    random.shuffle(patient_ids)
    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids = patient_ids[:n_val]
    if args.limit_patients:
        val_ids = val_ids[: args.limit_patients]
        print(f"SMOKE TEST: truncated to {len(val_ids)} patients")
    print(f"evaluating on {len(val_ids)} held-out patients")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "_smoke" if args.limit_patients else ""
    seed_tag = "" if args.seed == 0 else f"_seed{args.seed}"
    # Accumulation mode is part of the condition's identity, as it is for the window sweep. The
    # default stays unsuffixed so the three CSVs Section 7.12 published keep their paths and the
    # Otsu reproduction gate still resolves against them.
    wtag = "" if args.weighting == "uniform" else f"_{args.weighting}"
    csv_path = os.path.join(RESULTS_DIR,
                            f"rq13_{args.rq}{seed_tag}_thresholds{wtag}{suffix}_scores.csv")

    # Resume support. The gaussian read-out costs ~3.8x the uniform one per window, which puts the
    # scale-matched arms near this cluster's wall-clock cap, and a job killed at the wall would
    # otherwise discard hours of completed patients -- the same failure the preprocessing script hit
    # in Section 9 and was fixed the same way. Rows are appended per (patient, region) as they are
    # produced, and a restart skips whatever is already on disk.
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            done = {(row["patient_id"], row["region"]) for row in csv.DictReader(fh)}
        print(f"resuming: {len(done)} (patient, region) pairs already scored in {csv_path}")

    started = time.time()
    writer = None
    fh = open(csv_path, "a" if done else "w", newline="")
    incomplete = False
    rows = []          # rows written by *this* invocation; the resumed total is re-read at the end

    for region in ["ET", "TC", "WT"]:
        for i, pid in enumerate(val_ids):
            true_vol = float(true_volumes[pid][f"{region.lower()}_volume_mm3"])
            if true_vol <= 0:
                continue
            if (pid, region) in done:
                continue
            if args.max_hours and (time.time() - started) / 3600.0 > args.max_hours:
                print(f"\nreached --max_hours={args.max_hours}; stopping cleanly with work left",
                      flush=True)
                incomplete = True
                break

            data = np.load(os.path.join(PREPROC, f"{pid}.npz"))
            image = torch.from_numpy(data["image"]).float()
            gt_mask = region_mask(data["mask"], region)
            gt_voxels = int(gt_mask.sum())
            bin_label = size_bin(true_vol, region)

            heatmap = build_heatmap(model, image, text_proj, class_to_idx, region,
                                    scale_matched, device, weighting=args.weighting)

            # Tie-aware peak, for the same reason as RQ12: a strided sliding window makes the
            # heatmap piecewise-constant over blocks, so a bare argmax returns a block corner.
            dist_field = distance_transform_edt(~gt_mask, sampling=SPACING) if gt_voxels else None
            tied = np.array(np.unravel_index(
                np.flatnonzero(heatmap.ravel() == heatmap.max()), heatmap.shape)).T
            peak_idx = tuple(int(v) for v in tied[0])
            centroid = tuple(int(round(float(v))) for v in tied.mean(axis=0))
            argmax_hit, centroid_hit = bool(gt_mask[peak_idx]), bool(gt_mask[centroid])
            dist_mm = float(dist_field[peak_idx]) if dist_field is not None else float("nan")
            centroid_dist_mm = float(dist_field[centroid]) if dist_field is not None else float("nan")

            candidates = {"otsu": heatmap > otsu_threshold(heatmap.flatten())}
            for pct in PERCENTILES:
                candidates[f"pct{int(pct)}"] = top_k_mask(
                    heatmap, round(heatmap.size * (100.0 - pct) / 100.0))
            candidates["oracle_volume"] = top_k_mask(heatmap, gt_voxels)

            batch = []
            for method, pred_mask in candidates.items():
                dice, iou = dice_iou(pred_mask, gt_mask)
                batch.append({
                    "patient_id": pid, "region": region, "size_bin": bin_label,
                    "arm": args.rq, "threshold_method": method,
                    "true_volume_mm3": true_vol,
                    "gt_voxels_resized": gt_voxels,
                    "pred_voxels": int(pred_mask.sum()),
                    "pred_volume_mm3": int(pred_mask.sum()) * VOXEL_VOLUME_MM3,
                    "dice": dice, "iou": iou,
                    "argmax_hit": int(argmax_hit), "argmax_dist_mm": dist_mm,
                    "peak_tie_count": int(tied.shape[0]),
                    "centroid_hit": int(centroid_hit), "centroid_dist_mm": centroid_dist_mm,
                })
            # Append this (patient, region) before moving on, so a kill at the wall costs at most
            # the pair in flight rather than the whole run.
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(batch[0]))
                if not done:
                    writer.writeheader()
            writer.writerows(batch)
            fh.flush()
            rows.extend(batch)

            otsu_row = next(r for r in batch if r["threshold_method"] == "otsu")
            pct99_row = next(r for r in batch if r["threshold_method"] == "pct99")
            print(f"[{region}] ({i + 1}/{len(val_ids)}) {pid}: bin={bin_label} "
                  f"otsu={otsu_row['dice']:.3f} pct99={pct99_row['dice']:.3f}", flush=True)
        if incomplete:
            break

    fh.close()
    print(f"\nWrote {len(rows)} new rows to {csv_path}")

    if incomplete:
        print("EXITING 42: work remains, job script should resubmit")
        raise SystemExit(42)

    with open(csv_path, newline="") as f:
        allrows = list(csv.DictReader(f))
    print(f"\n=== {args.rq}: mean Dice by binarizer ({len(allrows)} rows total) ===")
    for m in ["otsu", "pct90", "pct95", "pct99", "oracle_volume"]:
        vals = [float(r["dice"]) for r in allrows if r["threshold_method"] == m]
        print(f"  {m:<15}{np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
