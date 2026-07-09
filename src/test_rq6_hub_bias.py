"""Re-runs the RQ4 noise probe on the RQ6 (uniformity-regularized) checkpoint, to check directly
whether the "large" embedding-space hub is actually gone, not just whether the uniformity loss
number went down."""
import os

import torch
import torch.nn.functional as F

from dataset_rq4 import CROP_SIZE_BY_BIN, MODEL_INPUT_SIZE
from model import TextVolumeAligner
from text_encoder import SIZE_CLASS_ORDER

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
TEXT_EMB_PATH = os.path.join(PREPROCESSED_DIR, "size_conditioned_text_embeddings.pt")
CKPT_PATH = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints/rq6_uniformity_aligner_best.pt"

N_SAMPLES_PER_BIN = 60


def make_noise_patch(crop_size, device):
    patch = torch.randn(1, 4, crop_size, crop_size, crop_size, device=device)
    if crop_size != MODEL_INPUT_SIZE:
        patch = F.interpolate(patch, size=(MODEL_INPUT_SIZE,) * 3, mode="trilinear", align_corners=False)
    return patch


def main():
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

    print("=== Pairwise cosine similarity among projected text classes ===")
    check_classes = ["ET_small", "ET_medium", "ET_large", "NONE"]
    for i, c1 in enumerate(check_classes):
        for c2 in check_classes[i + 1:]:
            sim = (text_proj[class_to_idx[c1]] @ text_proj[class_to_idx[c2]]).item()
            print(f"  {c1} vs {c2}: {sim:.4f}")

    region = "ET"
    bin_labels = ["small", "medium", "large"]
    torch.manual_seed(0)

    print("\n=== Cross-similarity check: does each noise pipeline prefer its OWN matching text over others? ===")
    with torch.no_grad():
        for bin_label in bin_labels:
            crop_size = CROP_SIZE_BY_BIN[bin_label]
            patches = torch.cat([make_noise_patch(crop_size, device) for _ in range(N_SAMPLES_PER_BIN)], dim=0)
            img_proj = model.encode_image(patches)
            sims = {b: (img_proj @ text_proj[class_to_idx[f"{region}_{b}"]]).mean().item() for b in bin_labels}
            best = max(sims, key=sims.get)
            match = "CORRECT (own label wins)" if best == bin_label else f"WRONG (prefers '{best}')"
            row = ", ".join(f"{b}={v:.4f}" for b, v in sims.items())
            print(f"noise via {bin_label}-pipeline (crop={crop_size}) -> similarity to: {row}  [{match}]")


if __name__ == "__main__":
    main()
