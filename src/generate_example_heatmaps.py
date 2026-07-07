"""Recompute and save ET-query heatmaps for two example patients (best-large, worst-small) for the
qualitative comparison figure."""
import os

import numpy as np
import torch

from localize import sliding_window_heatmap
from model import TextVolumeAligner
from text_encoder import REGION_ORDER

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
TEXT_EMB_PATH = os.path.join(PREPROCESSED_DIR, "subregion_text_embeddings.pt")
CKPT_PATH = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints/baseline_aligner.pt"
FIGURES_DIR = "/net/projects/ranalab/rajhansini/nlp_project/figures"

EXAMPLE_PATIENTS = {
    "large_example": "BraTS20_Training_225",
    "small_example": "BraTS20_Training_298",
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    text_embeds_dict = torch.load(TEXT_EMB_PATH, weights_only=True)
    text_embeds = torch.stack([text_embeds_dict[r] for r in REGION_ORDER])
    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        text_proj = model.encode_text(text_embeds.to(device))
    et_idx = REGION_ORDER.index("ET")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for tag, pid in EXAMPLE_PATIENTS.items():
        data = np.load(os.path.join(PREPROCESSED_DIR, f"{pid}.npz"))
        image = torch.from_numpy(data["image"]).float()
        seg = data["mask"]

        heatmap = sliding_window_heatmap(model, image, text_proj[et_idx], patch_size=32, stride=16, device=device).numpy()

        out_path = os.path.join(FIGURES_DIR, f"{tag}_{pid}.npz")
        np.savez_compressed(out_path, t1ce=data["image"][2], seg=seg, heatmap=heatmap)
        print(f"{tag} ({pid}): saved heatmap, shape={heatmap.shape}, max={heatmap.max():.3f}, min={heatmap.min():.3f}")


if __name__ == "__main__":
    main()
