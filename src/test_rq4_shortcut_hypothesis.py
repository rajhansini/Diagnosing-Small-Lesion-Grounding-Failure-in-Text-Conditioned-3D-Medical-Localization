"""Directly test the RQ4 shortcut-learning hypothesis: does the model key off resize-interpolation
artifacts (upsample blur vs downsample detail-loss) rather than genuine tumor content? Feed PURE
RANDOM NOISE through the same three crop-then-resize pipelines used in RQ4 training (16^3 upsampled,
32^3 native, 64^3 downsampled) -- there is no real content difference between these groups (all noise),
so if the model still assigns systematically different similarity to the matching size-text, that is
direct evidence it learned resize artifacts, not tumor features."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from dataset_rq4 import CROP_SIZE_BY_BIN, MODEL_INPUT_SIZE
from model import TextVolumeAligner
from text_encoder import SIZE_CLASS_ORDER

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
TEXT_EMB_PATH = os.path.join(PREPROCESSED_DIR, "size_conditioned_text_embeddings.pt")
CKPT_PATH = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints/rq4_scale_matched_aligner_best.pt"

N_SAMPLES_PER_BIN = 60


def make_noise_patch(crop_size, device):
    """Pure gaussian noise, same channel count as real data, run through the identical resize
    pipeline RQ4 training used (extract at crop_size, resize to MODEL_INPUT_SIZE)."""
    patch = torch.randn(1, 4, crop_size, crop_size, crop_size, device=device)
    if crop_size != MODEL_INPUT_SIZE:
        patch = F.interpolate(patch, size=(MODEL_INPUT_SIZE,) * 3, mode="trilinear", align_corners=False)
    return patch


def main():
    """Feed pure noise through RQ4's three resize pipelines to test for shortcut learning.

    If the model still assigns systematically different text-similarity scores to inputs containing
    zero tumor content, its apparent size-conditioning is reading resize-interpolation artifacts rather
    than anatomy. This is a direct test, not an inference from the localization numbers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    text_embeds_dict = torch.load(TEXT_EMB_PATH, weights_only=True)
    text_embeds = torch.stack([text_embeds_dict[c] for c in SIZE_CLASS_ORDER])
    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        text_proj = model.encode_text(text_embeds.to(device))
    class_to_idx = {c: i for i, c in enumerate(SIZE_CLASS_ORDER)}

    region = "ET"
    bin_labels = ["small", "medium", "large"]
    sims_by_bin = {b: [] for b in bin_labels}  # similarity of noise-through-that-pipeline to ITS OWN matching text

    torch.manual_seed(0)
    with torch.no_grad():
        for bin_label in bin_labels:
            crop_size = CROP_SIZE_BY_BIN[bin_label]
            text_idx = class_to_idx[f"{region}_{bin_label}"]
            for _ in range(N_SAMPLES_PER_BIN):
                patch = make_noise_patch(crop_size, device)
                img_proj = model.encode_image(patch)
                sim = (img_proj @ text_proj[text_idx]).item()
                sims_by_bin[bin_label].append(sim)

    print(f"\n=== Similarity of PURE NOISE (no real content) to its pipeline-matched '{region}_<bin>' text ===")
    print("(if these differ systematically across bins despite being pure noise, that's resize-artifact shortcut learning)")
    for b in bin_labels:
        vals = sims_by_bin[b]
        print(f"{region}_{b} (crop_size={CROP_SIZE_BY_BIN[b]}): mean_sim={np.mean(vals):.4f} std={np.std(vals):.4f}")

    small_vals, medium_vals, large_vals = sims_by_bin["small"], sims_by_bin["medium"], sims_by_bin["large"]
    f_stat, p_anova = stats.f_oneway(small_vals, medium_vals, large_vals)
    print(f"\nOne-way ANOVA across the three noise-pipeline groups: F={f_stat:.3f}, p={p_anova:.2e}")
    if p_anova < 0.05:
        print("=> SIGNIFICANT difference between groups using PURE NOISE. This directly confirms the model")
        print("   assigns systematically different scores based on resize pipeline alone, with zero real")
        print("   tumor content present -- supporting the resize-interpolation-artifact shortcut hypothesis.")
    else:
        print("=> No significant difference between noise groups -- does not support the shortcut hypothesis;")
        print("   the RQ4 failure likely has a different explanation.")

    # also check: does noise-through-small-pipeline get HIGHEST similarity to ITS OWN "small" text specifically,
    # compared to noise-through-small-pipeline's similarity to "medium"/"large" text (cross terms)?
    print("\n=== Cross-similarity check: does each noise pipeline prefer its OWN matching text over others? ===")
    with torch.no_grad():
        for bin_label in bin_labels:
            crop_size = CROP_SIZE_BY_BIN[bin_label]
            patches = torch.cat([make_noise_patch(crop_size, device) for _ in range(N_SAMPLES_PER_BIN)], dim=0)
            img_proj = model.encode_image(patches)
            row = []
            for other_bin in bin_labels:
                idx = class_to_idx[f"{region}_{other_bin}"]
                sim = (img_proj @ text_proj[idx]).mean().item()
                row.append(f"{other_bin}={sim:.4f}")
            print(f"noise via {bin_label}-pipeline (crop={crop_size}) -> similarity to: " + ", ".join(row))


if __name__ == "__main__":
    main()
