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
    """Assign a lesion to its small/medium/large tercile for one region.

    Cutoffs are per-region because the regions differ in absolute size by more than an order of
    magnitude (a "large" ET is smaller than a "small" WT), and are computed from true native-resolution
    volumes rather than the resampled grid so binning does not inherit resampling error.

    Args:
        volume_mm3: true lesion volume in mm^3 at native resolution.
        region: one of "ET", "TC", "WT".

    Returns:
        One of "small", "medium", "large".
    """
    lo, hi = SIZE_CUTOFFS[region]
    if volume_mm3 <= lo:
        return "small"
    if volume_mm3 <= hi:
        return "medium"
    return "large"


class BraTSSizeConditionedPatchDataset(Dataset):
    """Patch dataset for the RQ2 10-way size-conditioned classification task.

    Where BraTSPatchDataset labels a patch by region alone (4 classes), this labels it by region *and*
    the region's true size bin (ET_small ... WT_large, plus NONE = 10 classes), and the text side gets
    a matching size-conditioned description. RQ2 asks whether telling the model how big the target is
    supposed to be is enough to fix small-lesion localization.

    The physical crop stays a fixed patch_size regardless of the declared bin -- only the *text*
    changes. dataset_rq4.BraTSScaleMatchedPatchDataset also matches the crop size, which is what
    separates the language manipulation from the receptive-field one.

    Args:
        preprocessed_dir: directory of per-patient .npz files.
        lesion_csv_path: lesion_volumes.csv, source of true per-region volumes.
        patient_ids: patient IDs to include; the caller owns the train/val split.
        patch_size: edge length of the cubic patch, identical for every size bin.
    """
    def __init__(self, preprocessed_dir, lesion_csv_path, patient_ids, patch_size=32):
        """Build the (patient, region, size-class) index from the lesion-volume table."""
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
        """Number of (patient, region, size-class) entries available."""
        return len(self.index)

    def __getitem__(self, idx):
        """Sample one patch and return it with its 10-way size-conditioned class index.

        Args:
            idx: position in the index.

        Returns:
            (patch, label) with patch a (4, p, p, p) float tensor and label indexing SIZE_CLASS_ORDER.
        """
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
