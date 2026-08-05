"""Sliding-window text-conditioned localization: scores overlapping windows against a text query
embedding and accumulates them into a per-voxel heatmap over the full volume."""
import torch
import torch.nn.functional as F


def sliding_window_heatmap(model, volume, text_proj_vec, patch_size=32, stride=16, device="cpu",
                            batch_size=64, window_size=None, model_input_size=32):
    """Sweep the trained patch encoder across a volume and accumulate per-voxel query similarity.

    This replaced Grad-CAM, which is unusable here: the encoder global-average-pools each patch to a
    single embedding, leaving no spatial feature map near the output to back-propagate onto. Sweeping
    the encoder instead measures space the way the model actually represents it.

    Every window contributes its scalar score to all voxels it covers, and the result is divided by
    the per-voxel coverage count, so overlapping strides produce a smooth average rather than a sum
    biased toward the volume's interior.

    Args:
        model: a trained TextVolumeAligner.
        volume: (4, D, H, W) float tensor, one preprocessed multi-modal patient volume.
        text_proj_vec: (proj_dim,) already projected and L2-normalized text query embedding.
        patch_size: default physical window size, used when window_size is None.
        stride: step between consecutive window origins along each axis.
        device: torch device to run the encoder on.
        batch_size: number of windows encoded per forward pass.
        window_size: physical size of the window swept over the volume. If it differs from
            model_input_size, each extracted window is trilinearly resized to model_input_size before
            encoding, which lets a model trained at one patch size be evaluated at a different
            receptive-field scale without retraining (the RQ3/RQ3b/RQ3c multi-scale sweep).
        model_input_size: the patch edge length the model was actually trained on.

    Returns:
        (D, H, W) float tensor of averaged cosine similarities to the query.
    """
    if window_size is None:
        window_size = patch_size
    D, H, W = volume.shape[1:]
    heatmap = torch.zeros(D, H, W)
    counts = torch.zeros(D, H, W)

    def starts(dim):
        """Window origins along one axis, always including a final flush-right window.

        Without the explicit last element, an axis length that is not a whole number of strides past
        the window would leave a strip at the far edge never covered by any window, and those voxels
        would keep a count of zero.
        """
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
