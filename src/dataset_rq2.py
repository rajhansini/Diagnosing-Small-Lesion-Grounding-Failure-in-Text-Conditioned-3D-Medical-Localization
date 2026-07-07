"""Size-conditioned patch dataset for RQ2: labels each ET/TC/WT patch with its region's true size bin."""
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import region_mask
from text_encoder import SIZE_CLASS_ORDER

SIZE_CLASS_TO_IDX = {c: i for i, c in enumerate(SIZE_CLASS_ORDER)}

# same tercile cutoffs used throughout the project (computed from the full 369-patient lesion_volumes.csv)
SIZE_CUTOFFS = {
    "ET": (9050.6, 26245.3),
    "TC": (19190.7, 49948.2),
    "WT": (63126.2, 125309.1),
}


def size_bin(volume_mm3, region):
    lo, hi = SIZE_CUTOFFS[region]
    if volume_mm3 <= lo:
        return "small"
    if volume_mm3 <= hi:
        return "medium"
    return "large"


class BraTSSizeConditionedPatchDataset(Dataset):
    def __init__(self, preprocessed_dir, lesion_csv_path, patient_ids, patch_size=32):
        self.preprocessed_dir = preprocessed_dir
        self.patch_size = patch_size
        with open(lesion_csv_path, newline="") as f:
            self.true_volumes = {row["patient_id"]: row for row in csv.DictReader(f)}

        self.index = []  # list of (patient_id, region, class_name)
        for pid in patient_ids:
            seg = np.load(os.path.join(preprocessed_dir, f"{pid}.npz"))["mask"]
            for region in ["ET", "TC", "WT"]:
                if not region_mask(seg, region).any():
                    continue
                true_vol = float(self.true_volumes[pid][f"{region.lower()}_volume_mm3"])
                bin_label = size_bin(true_vol, region)
                self.index.append((pid, region, f"{region}_{bin_label}"))
            self.index.append((pid, "NONE", "NONE"))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pid, region, class_name = self.index[idx]
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

        return torch.from_numpy(patch.copy()).float(), SIZE_CLASS_TO_IDX[class_name]
