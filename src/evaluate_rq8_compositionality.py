"""RQ8: does the trained model use language compositionally, or as a class-ID lookup?

Evaluation-only probe on the frozen RQ1 baseline -- no retraining. For every held-out patient and
region we compute the sliding-window heatmap under five text queries (original / negated / shuffled
/ swapped / generic, built by build_rq8_probe_texts.py) using the identical 32^3 stride-16 protocol,
and measure three things per condition:

  1. cosine similarity to the original query in the model's *projected* space (post text_proj),
     i.e. how far apart the trained model considers these queries to be;
  2. Spearman correlation between the resulting heatmap and the original query's heatmap, i.e.
     how far apart the model's spatial *behaviour* actually is;
  3. Dice under the standard Otsu protocol, i.e. whether localization quality changes at all.

The 'original' condition must reproduce results/rq1_localization_scores.csv exactly. That is an
end-to-end correctness gate on this script, and it is checked automatically at the end of the run.

The diagnostic reading: negation preserves nearly all content words while inverting the clinical
meaning, and shuffling preserves the bag of words exactly while destroying syntax. If a model
genuinely composes meaning, both should move the heatmap substantially. If heatmap correlations
stay near 1.0, the text query is functioning as an opaque class identifier -- which is the
mechanistic version of this project's "architecture, not language" claim.
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
from scipy import stats

from dataset import region_mask
from evaluate_rq1 import dice_iou, load_true_volumes, otsu_threshold, size_bin
from localize import sliding_window_heatmap
from model import TextVolumeAligner
from text_encoder import REGION_ORDER

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
PROBE_EMB_PATH = os.path.join(PREPROCESSED_DIR, "rq8_probe_text_embeddings.pt")
CKPT_PATH = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints/baseline_aligner.pt"
RESULTS_DIR = "/net/projects/ranalab/rajhansini/nlp_project/results"
RQ1_CSV = os.path.join(RESULTS_DIR, "rq1_localization_scores.csv")

CONDITIONS = ["original", "negated", "shuffled", "swapped", "generic"]
REGIONS = ["ET", "TC", "WT"]


def main():
    """Score every compositionality probe condition and verify 'original' reproduces RQ1 exactly."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_patients", type=int, default=None,
                        help="smoke tests only: truncate the val set AFTER the split is computed. "
                             "Results are written to a _smoke CSV.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RQ8 compositionality probe | device={device} ===")

    true_volumes = load_true_volumes()
    probes = torch.load(PROBE_EMB_PATH, weights_only=True)

    text_dim = probes[f"original|{REGION_ORDER[0]}"].shape[0]
    model = TextVolumeAligner(text_dim=text_dim).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"loaded frozen RQ1 baseline from {CKPT_PATH}")

    # Project every probe query through the model's trained text head.
    proj = {}
    with torch.no_grad():
        for key, emb in probes.items():
            proj[key] = model.encode_text(emb.unsqueeze(0).to(device))[0]

    print("\n=== Projected-space cosine similarity to the ORIGINAL query ===")
    print("(1.0 means the trained model cannot tell the two queries apart at all)")
    print(f"{'condition':<12}" + "".join(f"{r:>10}" for r in REGIONS))
    for condition in CONDITIONS:
        line = f"{condition:<12}"
        for region in REGIONS:
            sim = torch.dot(proj[f"original|{region}"], proj[f"{condition}|{region}"]).item()
            line += f"{sim:>10.4f}"
        print(line)

    random.seed(0)
    patient_ids = sorted(f[:-4] for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz"))
    random.shuffle(patient_ids)
    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids = patient_ids[:n_val]
    if args.limit_patients:
        val_ids = val_ids[: args.limit_patients]
        print(f"SMOKE TEST: truncated to {len(val_ids)} validation patients")
    print(f"\nevaluating on {len(val_ids)} held-out validation patients (seed 0 split, identical to RQ1)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for region in REGIONS:
        for i, pid in enumerate(val_ids):
            true_vol = float(true_volumes[pid][f"{region.lower()}_volume_mm3"])
            if true_vol <= 0:
                continue

            data = np.load(os.path.join(PREPROCESSED_DIR, f"{pid}.npz"))
            image = torch.from_numpy(data["image"]).float()
            gt_mask = region_mask(data["mask"], region)
            bin_label = size_bin(true_vol, region)

            heatmaps = {}
            for condition in CONDITIONS:
                heatmaps[condition] = sliding_window_heatmap(
                    model, image, proj[f"{condition}|{region}"], patch_size=32, stride=16, device=device
                ).numpy()

            reference = heatmaps["original"].ravel()
            for condition in CONDITIONS:
                heatmap = heatmaps[condition]
                pred_mask = heatmap > otsu_threshold(heatmap.flatten())
                dice, iou = dice_iou(pred_mask, gt_mask)
                rho = 1.0 if condition == "original" else float(stats.spearmanr(reference, heatmap.ravel()).statistic)
                rows.append({
                    "patient_id": pid, "region": region, "size_bin": bin_label,
                    "condition": condition, "true_volume_mm3": true_vol,
                    "dice": dice, "iou": iou,
                    "heatmap_spearman_vs_original": rho,
                    "proj_cosine_vs_original": torch.dot(
                        proj[f"original|{region}"], proj[f"{condition}|{region}"]).item(),
                })

            summary = " ".join(f"{c[:4]}={next(r['dice'] for r in rows[-len(CONDITIONS):] if r['condition'] == c):.3f}"
                               for c in CONDITIONS)
            print(f"[{region}] ({i + 1}/{len(val_ids)}) {pid}: bin={bin_label} dice -> {summary}")

    fieldnames = ["patient_id", "region", "size_bin", "condition", "true_volume_mm3", "dice", "iou",
                  "heatmap_spearman_vs_original", "proj_cosine_vs_original"]
    csv_path = os.path.join(RESULTS_DIR, "rq8_compositionality_scores%s.csv" % ("_smoke" if args.limit_patients else ""))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {csv_path}")

    print("\n=== Heatmap Spearman correlation vs. the original query ===")
    print("(near 1.0 = the model's spatial behaviour is unchanged by the linguistic manipulation)")
    print(f"{'condition':<12}" + "".join(f"{r:>18}" for r in REGIONS))
    for condition in CONDITIONS:
        line = f"{condition:<12}"
        for region in REGIONS:
            vals = [r["heatmap_spearman_vs_original"] for r in rows if r["region"] == region and r["condition"] == condition]
            line += f"{np.mean(vals):>12.4f} +-{np.std(vals):<4.3f}" if vals else f"{'':>18}"
        print(line)

    print("\n=== Mean Dice by condition (pooled over size bins) ===")
    print(f"{'condition':<12}" + "".join(f"{r:>10}" for r in REGIONS))
    for condition in CONDITIONS:
        line = f"{condition:<12}"
        for region in REGIONS:
            vals = [r["dice"] for r in rows if r["region"] == region and r["condition"] == condition]
            line += f"{np.mean(vals):>10.4f}" if vals else f"{'':>10}"
        print(line)

    print("\n=== Paired Wilcoxon: each condition vs. original, per region ===")
    for condition in CONDITIONS[1:]:
        for region in REGIONS:
            pairs = {}
            for r in rows:
                if r["region"] == region and r["condition"] in ("original", condition):
                    pairs.setdefault(r["patient_id"], {})[r["condition"]] = r["dice"]
            both = [(v["original"], v[condition]) for v in pairs.values() if len(v) == 2]
            if len(both) < 5:
                continue
            orig, other = zip(*both)
            if np.allclose(orig, other):
                print(f"  {condition:<10} {region}: n={len(both)} IDENTICAL Dice to original (p undefined)")
                continue
            p = stats.wilcoxon(orig, other).pvalue
            print(f"  {condition:<10} {region}: n={len(both)} mean {np.mean(orig):.4f} -> {np.mean(other):.4f} "
                  f"(delta {np.mean(other) - np.mean(orig):+.4f}), p={p:.4g}")

    # ---- End-to-end correctness gate: 'original' must reproduce RQ1 exactly ----
    print("\n=== Correctness gate: 'original' condition vs. results/rq1_localization_scores.csv ===")
    with open(RQ1_CSV, newline="") as f:
        rq1 = {(r["patient_id"], r["region"]): float(r["dice"]) for r in csv.DictReader(f)}
    diffs = [abs(r["dice"] - rq1[(r["patient_id"], r["region"])])
             for r in rows if r["condition"] == "original" and (r["patient_id"], r["region"]) in rq1]
    if diffs:
        print(f"  compared {len(diffs)} patient-region pairs | max |diff| = {max(diffs):.2e}")
        print("  PASS: reproduces RQ1" if max(diffs) < 1e-6 else
              f"  WARNING: does not reproduce RQ1 (max diff {max(diffs):.2e}) -- investigate before trusting this run")
    else:
        print("  WARNING: no overlapping rows found against RQ1")


if __name__ == "__main__":
    main()
