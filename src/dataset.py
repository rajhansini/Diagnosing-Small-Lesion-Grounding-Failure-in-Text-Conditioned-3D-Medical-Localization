"""Patch-level dataset: samples one labeled 3D patch per (patient, region) pair for contrastive training."""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from text_encoder import REGION_ORDER

REGION_TO_IDX = {r: i for i, r in enumerate(REGION_ORDER)}  # ET=0, TC=1, WT=2, NONE=3


def region_mask(seg, region):
    """Convert a raw BraTS label volume into a boolean mask for one evaluation region.

    BraTS ships per-voxel labels (1=necrotic/non-enhancing core, 2=peritumoral edema, 4=enhancing
    tumor; 0=background). The three regions the challenge actually scores are nested unions of those
    labels rather than the labels themselves, which is why this indirection exists. NONE is the
    complement of WT and supplies the negative class for contrastive training.

    Args:
        seg: integer label volume with BraTS codes.
        region: one of "ET", "TC", "WT", "NONE".

    Returns:
        Boolean np.ndarray with the same shape as seg.

    Raises:
        ValueError: if region is not one of the four recognised names.
    """
    if region == "ET":
        return seg == 4
    if region == "TC":
        return (seg == 1) | (seg == 4)
    if region == "WT":
        return (seg == 1) | (seg == 2) | (seg == 4)
    if region == "NONE":
        return ~((seg == 1) | (seg == 2) | (seg == 4))
    raise ValueError(region)


class BraTSPatchDataset(Dataset):
    """Serves one randomly-placed labeled patch per (patient, region) pair.

    The index is built once at construction by checking which regions are actually present in each
    patient's segmentation, so patients with no enhancing tumor (27 of 369 in BraTS2020) contribute no
    ET example rather than an empty one. Patch placement is re-randomized on every __getitem__, so
    across epochs the model sees many different crops of the same region -- the dataset length is the
    number of (patient, region) pairs, not the number of distinct patches.

    Args:
        preprocessed_dir: directory of per-patient .npz files written by preprocess.py.
        patient_ids: patient IDs to include; the caller owns the train/val split.
        patch_size: edge length of the cubic patch to sample.
    """

    def __init__(self, preprocessed_dir, patient_ids, patch_size=32):
        """Build the (patient, region) index, skipping regions absent from a patient's segmentation."""
        self.preprocessed_dir = preprocessed_dir
        self.patch_size = patch_size
        self.index = []
        for pid in patient_ids:
            seg = np.load(os.path.join(preprocessed_dir, f"{pid}.npz"))["mask"]
            for region in REGION_ORDER:
                if region_mask(seg, region).any():
                    self.index.append((pid, region))

    def __len__(self):
        """Number of (patient, region) pairs available."""
        return len(self.index)

    def __getitem__(self, idx):
        """Sample one patch centered on a random voxel of the requested region.

        The centre is clamped so the patch stays inside the volume, and any residual shortfall at the
        border is zero-padded, so the returned tensor always has the full patch shape.

        Args:
            idx: position in the (patient, region) index.

        Returns:
            (patch, label) where patch is a (4, p, p, p) float tensor and label is the region's
            integer class index (ET=0, TC=1, WT=2, NONE=3).
        """
        pid, region = self.index[idx]
        data = np.load(os.path.join(self.preprocessed_dir, f"{pid}.npz"))
        image, seg = data["image"], data["mask"]
        mask = region_mask(seg, region)
        voxel_coords = np.argwhere(mask)
        center = voxel_coords[random.randrange(len(voxel_coords))]

        p = self.patch_size
        dims = seg.shape
        lo = [max(0, min(int(c) - p // 2, d - p)) for c, d in zip(center, dims)]
        patch = image[:, lo[0]:lo[0] + p, lo[1]:lo[1] + p, lo[2]:lo[2] + p]

        pad = [(0, 0)] + [(0, p - patch.shape[i + 1]) for i in range(3)]
        if any(pd[1] > 0 for pd in pad):
            patch = np.pad(patch, pad, mode="constant")

        return torch.from_numpy(patch.copy()).float(), REGION_TO_IDX[region]
