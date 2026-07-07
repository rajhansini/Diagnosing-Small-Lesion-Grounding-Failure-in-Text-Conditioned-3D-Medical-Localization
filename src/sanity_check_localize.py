"""Quick sanity check: does the sliding-window heatmap actually score higher inside the true lesion than outside?"""
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    text_embeds_dict = torch.load(TEXT_EMB_PATH, weights_only=True)
    text_embeds = torch.stack([text_embeds_dict[r] for r in REGION_ORDER])

    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        text_proj = model.encode_text(text_embeds.to(device))  # (4, proj_dim), normalized

    # same train/val split logic as train_baseline.py, so this is a held-out val patient
    random.seed(0)
    patient_ids = sorted(f[:-4] for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz"))
    random.shuffle(patient_ids)
    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids = patient_ids[:n_val]

    # find a val patient with a reasonably sized ET region for a clear test
    test_pid = None
    for pid in val_ids:
        seg = np.load(os.path.join(PREPROCESSED_DIR, f"{pid}.npz"))["mask"]
        if (seg == 4).sum() > 500:
            test_pid = pid
            break
    print("test patient:", test_pid)

    data = np.load(os.path.join(PREPROCESSED_DIR, f"{test_pid}.npz"))
    image = torch.from_numpy(data["image"]).float()
    seg = data["mask"]

    et_idx = REGION_ORDER.index("ET")
    heatmap = sliding_window_heatmap(model, image, text_proj[et_idx], patch_size=32, stride=16, device=device)
    heatmap = heatmap.numpy()

    et_mask = region_mask(seg, "ET")
    inside = heatmap[et_mask]
    outside = heatmap[~et_mask]
    print(f"ET query heatmap: mean score inside true ET region = {inside.mean():.4f} (n={inside.size})")
    print(f"ET query heatmap: mean score outside true ET region = {outside.mean():.4f} (n={outside.size})")
    print(f"difference (inside - outside): {inside.mean() - outside.mean():.4f}")

    # also check the NONE query as a contrast: should score LOWER inside the ET region than outside
    none_idx = REGION_ORDER.index("NONE")
    heatmap_none = sliding_window_heatmap(model, image, text_proj[none_idx], patch_size=32, stride=16, device=device).numpy()
    print(f"\nNONE query heatmap: mean score inside true ET region = {heatmap_none[et_mask].mean():.4f}")
    print(f"NONE query heatmap: mean score outside true ET region = {heatmap_none[~et_mask].mean():.4f}")


if __name__ == "__main__":
    main()
