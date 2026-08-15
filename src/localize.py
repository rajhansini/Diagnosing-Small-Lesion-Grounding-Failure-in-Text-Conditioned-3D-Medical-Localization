"""Sliding-window text-conditioned localization: scores overlapping windows against a text query
embedding and accumulates them into a per-voxel heatmap over the full volume."""
import torch
import torch.nn.functional as F


def _gaussian_kernel(size, sigma_frac=0.25):
    """Separable 3D Gaussian over a window, peaked at its centre, normalized to a maximum of 1.

    Used by the "gaussian" accumulation mode. The width scales with the window so the shape of the
    weighting is the same at every window size, which is what makes conditions comparable.

    Args:
        size: window edge length in voxels.
        sigma_frac: standard deviation as a fraction of the window edge.

    Returns:
        (size, size, size) float tensor.
    """
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-coords.pow(2) / (2.0 * (sigma_frac * size) ** 2))
    k = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return k / k.max()


def sliding_window_heatmap(model, volume, text_proj_vec, patch_size=32, stride=16, device="cpu",
                            batch_size=64, window_size=None, model_input_size=32,
                            weighting="uniform", sigma_frac=0.25):
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
        weighting: how a window's single scalar score is spread over the voxels it covers.
            "uniform" (default, and what every result before RQ14 used) adds the score equally to
            all window_size^3 voxels, which makes the heatmap piecewise-constant over stride^3
            blocks and is the resolution bottleneck Section 6.3 measures. "gaussian" weights the
            contribution by a centre-peaked kernel instead, so a window's evidence is attributed
            mostly to where it was looking, and overlapping windows can disagree within a block.
        sigma_frac: for the gaussian mode, the kernel's standard deviation as a fraction of the
            window edge. Scaling with the window rather than fixing it in voxels keeps the shape of
            the weighting identical at every window size, which is what makes conditions comparable.
            Too wide approaches uniform smearing; too narrow discards the window's real extent.

    Returns:
        (D, H, W) float tensor of averaged cosine similarities to the query.
    """
    if window_size is None:
        window_size = patch_size
    if weighting not in ("uniform", "gaussian"):
        raise ValueError(f"unknown weighting {weighting!r}; expected 'uniform' or 'gaussian'")
    kernel = _gaussian_kernel(window_size, sigma_frac) if weighting == "gaussian" else None
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
                sl = (slice(d, d + window_size), slice(h, h + window_size), slice(w, w + window_size))
                if kernel is None:
                    heatmap[sl] += s
                    counts[sl] += 1
                else:
                    # Weighted mean rather than weighted sum: the kernel accumulates into counts as
                    # well, so dividing at the end still yields an average of the covering windows'
                    # scores, just one that trusts each window most near its own centre.
                    heatmap[sl] += s * kernel
                    counts[sl] += kernel

    return heatmap / counts.clamp(min=1)
