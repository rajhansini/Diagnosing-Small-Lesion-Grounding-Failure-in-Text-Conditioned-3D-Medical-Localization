"""RQ4 dataset: like RQ2's size-conditioned patches, but the physical crop size is matched to the
lesion's true size bin (small/medium/large -> 16/32/64 voxels), then resized to the model's canonical
32^3 input. Tests whether matching training-time receptive field to declared size (not just the text)
lets size-conditioning actually work."""
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset import region_mask
from dataset_rq2 import size_bin
from text_encoder import SIZE_CLASS_ORDER

SIZE_CLASS_TO_IDX = {c: i for i, c in enumerate(SIZE_CLASS_ORDER)}
CROP_SIZE_BY_BIN = {"small": 16, "medium": 32, "large": 64}
MODEL_INPUT_SIZE = 32


class BraTSScaleMatchedPatchDataset(Dataset):
    """Size-conditioned patch dataset whose physical crop size is matched to the true size bin.

    Small/medium/large lesions are cropped at 16/32/64 voxels respectively and then resized to the
    model's canonical 32^3 input, so the network always sees a target that fills a similar fraction of
    its receptive field. RQ2 changed only the text and did not fix small lesions; this changes the
    geometry too, testing whether the bottleneck was receptive field rather than language.

    The resize is also what makes this arm vulnerable to shortcut learning: three crop sizes mean
    three distinct interpolation signatures, and test_rq4_shortcut_hypothesis.py confirms the trained
    model keys off those artifacts rather than tumor content.

    Args:
        preprocessed_dir: directory of per-patient .npz files.
        lesion_csv_path: lesion_volumes.csv, source of true per-region volumes.
        patient_ids: patient IDs to include; the caller owns the train/val split.
    """
    def __init__(self, preprocessed_dir, lesion_csv_path, patient_ids):
        """Build the index, recording each entry's bin-matched physical crop size."""
        self.preprocessed_dir = preprocessed_dir
        with open(lesion_csv_path, newline="") as f:
            self.true_volumes = {row["patient_id"]: row for row in csv.DictReader(f)}

        self.index = []  # list of (patient_id, region, class_name, crop_size)
        for pid in patient_ids:
            seg = np.load(os.path.join(preprocessed_dir, f"{pid}.npz"))["mask"]
            for region in ["ET", "TC", "WT"]:
                if not region_mask(seg, region).any():
                    continue
                true_vol = float(self.true_volumes[pid][f"{region.lower()}_volume_mm3"])
                bin_label = size_bin(true_vol, region)
                self.index.append((pid, region, f"{region}_{bin_label}", CROP_SIZE_BY_BIN[bin_label]))
            self.index.append((pid, "NONE", "NONE", MODEL_INPUT_SIZE))

    def __len__(self):
        """Number of (patient, region, size-class) entries available."""
        return len(self.index)

    def __getitem__(self, idx):
        """Sample a bin-matched crop, resize it to the model's input size, and label it.

        Args:
            idx: position in the index.

        Returns:
            (patch, label) with patch a (4, 32, 32, 32) float tensor after trilinear resizing.
        """
        pid, region, class_name, crop_size = self.index[idx]
        data = np.load(os.path.join(self.preprocessed_dir, f"{pid}.npz"))
        image, seg = data["image"], data["mask"]
        mask = region_mask(seg, region)
        voxel_coords = np.argwhere(mask)
        center = voxel_coords[random.randrange(len(voxel_coords))]

        p = crop_size
        dims = seg.shape
        lo = [max(0, min(int(c) - p // 2, d - p)) for c, d in zip(center, dims)]
        patch = image[:, lo[0]:lo[0] + p, lo[1]:lo[1] + p, lo[2]:lo[2] + p]

        pad = [(0, 0)] + [(0, p - patch.shape[i + 1]) for i in range(3)]
        if any(pd[1] > 0 for pd in pad):
            patch = np.pad(patch, pad, mode="constant")

        patch_t = torch.from_numpy(patch.copy()).float()
        if crop_size != MODEL_INPUT_SIZE:
            patch_t = F.interpolate(patch_t.unsqueeze(0), size=(MODEL_INPUT_SIZE,) * 3, mode="trilinear", align_corners=False).squeeze(0)

        return patch_t, SIZE_CLASS_TO_IDX[class_name]
