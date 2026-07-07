"""Sliding-window text-conditioned localization: scores overlapping windows against a text query
embedding and accumulates them into a per-voxel heatmap over the full volume."""
import torch
import torch.nn.functional as F


def sliding_window_heatmap(model, volume, text_proj_vec, patch_size=32, stride=16, device="cpu",
                            batch_size=64, window_size=None, model_input_size=32):
    """volume: (4, D, H, W) tensor. text_proj_vec: (proj_dim,) already projected+normalized text embedding.
    window_size: physical size of the window swept over the volume (defaults to patch_size, i.e. same
    behavior as before). If window_size != model_input_size, each extracted window is resized to
    model_input_size before encoding -- this lets a model trained at one patch size be evaluated at a
    different receptive-field scale without retraining (RQ3 multi-scale sweep).
    Returns a (D, H, W) tensor of averaged cosine-similarity scores from all windows covering each voxel."""
    if window_size is None:
        window_size = patch_size
    D, H, W = volume.shape[1:]
    heatmap = torch.zeros(D, H, W)
    counts = torch.zeros(D, H, W)

    def starts(dim):
        s = list(range(0, dim - window_size + 1, stride))
        if not s or s[-1] != dim - window_size:
            s.append(dim - window_size)
        return s

    coords = [(d, h, w) for d in starts(D) for h in starts(H) for w in starts(W)]

    model.eval()
    with torch.no_grad():
        text_proj_vec = text_proj_vec.to(device)
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = torch.stack([
                volume[:, d:d + window_size, h:h + window_size, w:w + window_size] for d, h, w in batch_coords
            ]).to(device)
            if window_size != model_input_size:
                patches = F.interpolate(patches, size=(model_input_size,) * 3, mode="trilinear", align_corners=False)
            img_proj = model.encode_image(patches)  # (B, proj_dim), L2-normalized
            scores = (img_proj @ text_proj_vec).cpu()  # (B,) cosine similarity
            for (d, h, w), s in zip(batch_coords, scores):
                heatmap[d:d + window_size, h:h + window_size, w:w + window_size] += s
                counts[d:d + window_size, h:h + window_size, w:w + window_size] += 1

    return heatmap / counts.clamp(min=1)
