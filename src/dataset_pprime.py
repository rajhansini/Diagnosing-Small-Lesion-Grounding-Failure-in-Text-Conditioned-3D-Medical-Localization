"""Full-volume segmentation dataset for the P' supervised reference point.

Every other dataset in this project samples small labeled *patches* for contrastive alignment. This
one is different on purpose: it serves whole preprocessed volumes with dense multi-label voxel
targets, which is the input format a conventional supervised segmentation network expects. It exists
so the P' check (src/train_pprime_supervised.py) exercises the same preprocessed data, the same
train/val split, and the same region definitions as the text-conditioned experiments, while solving
the previously-studied problem (BraTS tumor segmentation) that published Dice numbers exist for.

Targets are the three standard BraTS evaluation regions as a 3-channel multi-label mask (ET/TC/WT),
not a 4-way mutually-exclusive labelling -- the regions are nested (ET subset of TC subset of WT), so
multi-label sigmoid is the formulation the BraTS challenge itself scores against.
"""
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import region_mask

SEG_REGIONS = ["ET", "TC", "WT"]


def stack_region_masks(seg):
    """Build the (3, D, H, W) float multi-label target from a raw BraTS label volume.

    Channels are ET/TC/WT in that order, matching SEG_REGIONS and the region ordering used by every
    evaluation script in this project.

    Args:
        seg: integer label volume with BraTS codes (1=necrotic, 2=edema, 4=enhancing).

    Returns:
        np.ndarray of shape (3, D, H, W), dtype float32, values in {0.0, 1.0}.
    """
    return np.stack([region_mask(seg, r) for r in SEG_REGIONS]).astype(np.float32)


class BraTSSegDataset(Dataset):
    """Serves (image, multi-label mask) pairs for supervised segmentation training.

    In training mode it returns a random `crop_size` cube per sample, biased toward tumor-containing
    locations so that batches are not dominated by empty background (the whole tumor occupies only a
    few percent of a skull-stripped 128^3 brain volume, so uniform cropping would make most batches
    label-free and stall Dice loss early in training). In eval mode it returns the full volume, since
    the network is fully convolutional and inference is done in one forward pass.

    Args:
        preprocessed_dir: directory of per-patient .npz files written by preprocess.py.
        patient_ids: patient IDs to serve; caller is responsible for the train/val split.
        crop_size: edge length of the random training cube. Ignored when train=False.
        train: True for random biased crops, False for full-volume evaluation.
        fg_prob: probability that a training crop is centered on a whole-tumor voxel.
    """

    def __init__(self, preprocessed_dir, patient_ids, crop_size=96, train=True, fg_prob=0.7):
        """Store the patient list and cropping configuration."""
        self.preprocessed_dir = preprocessed_dir
        self.patient_ids = list(patient_ids)
        self.crop_size = crop_size
        self.train = train
        self.fg_prob = fg_prob

    def __len__(self):
        """Number of patients served (one sample per patient per epoch)."""
        return len(self.patient_ids)

    def _random_crop_origin(self, wt_mask, dims):
        """Pick the lower corner of a training crop, usually centered on tumor.

        Args:
            wt_mask: boolean whole-tumor mask, used to find foreground voxels.
            dims: (D, H, W) of the full volume.

        Returns:
            list of 3 ints, the crop's lower corner, clamped to stay inside the volume.
        """
        c = self.crop_size
        fg = np.argwhere(wt_mask)
        if len(fg) > 0 and random.random() < self.fg_prob:
            center = fg[random.randrange(len(fg))]
            return [int(np.clip(int(ctr) - c // 2, 0, d - c)) for ctr, d in zip(center, dims)]
        return [random.randint(0, d - c) for d in dims]

    def __getitem__(self, idx):
        """Return one (image, target) pair as float tensors.

        Shapes are (4, c, c, c) / (3, c, c, c) in training mode and (4, D, H, W) / (3, D, H, W) in
        eval mode.
        """
        pid = self.patient_ids[idx]
        data = np.load(os.path.join(self.preprocessed_dir, f"{pid}.npz"))
        image, seg = data["image"], data["mask"]
        target = stack_region_masks(seg)

        if not self.train:
            return torch.from_numpy(image.copy()).float(), torch.from_numpy(target.copy()).float()

        c = self.crop_size
        lo = self._random_crop_origin(target[SEG_REGIONS.index("WT")] > 0, seg.shape)
        image = image[:, lo[0]:lo[0] + c, lo[1]:lo[1] + c, lo[2]:lo[2] + c]
        target = target[:, lo[0]:lo[0] + c, lo[1]:lo[1] + c, lo[2]:lo[2] + c]
        return torch.from_numpy(image.copy()).float(), torch.from_numpy(target.copy()).float()
