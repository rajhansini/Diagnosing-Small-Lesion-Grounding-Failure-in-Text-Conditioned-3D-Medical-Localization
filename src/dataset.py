"""Patch-level dataset: samples one labeled 3D patch per (patient, region) pair for contrastive training."""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from text_encoder import REGION_ORDER

REGION_TO_IDX = {r: i for i, r in enumerate(REGION_ORDER)}  # ET=0, TC=1, WT=2, NONE=3


def region_mask(seg, region):
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
    def __init__(self, preprocessed_dir, patient_ids, patch_size=32):
        self.preprocessed_dir = preprocessed_dir
        self.patch_size = patch_size
        self.index = []
        for pid in patient_ids:
            seg = np.load(os.path.join(preprocessed_dir, f"{pid}.npz"))["mask"]
            for region in REGION_ORDER:
                if region_mask(seg, region).any():
                    self.index.append((pid, region))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
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
