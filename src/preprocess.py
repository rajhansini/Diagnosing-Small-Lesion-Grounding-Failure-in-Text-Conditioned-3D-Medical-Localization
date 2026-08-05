"""Preprocess BraTS2020: per-modality normalization, resize to a fixed grid, and per-patient lesion-volume stats."""
import argparse
import csv
import glob
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

MODALITIES = ["flair", "t1", "t1ce", "t2"]

DEFAULT_DATA_ROOT = "/net/projects/ranalab/rajhansini/nlp_project/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
DEFAULT_OUT_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"


def load_patient(patient_dir, patient_id):
    """Load one patient's four MRI modalities and segmentation from disk.

    Handles the known BraTS2020 quirk where patient 355's segmentation ships under a non-standard
    filename (W39_1998.09.19_Segm.nii), a leftover from the original hospital anonymization that was
    never corrected in this release.

    Args:
        patient_dir: directory containing that patient's NIfTI files.
        patient_id: e.g. "BraTS20_Training_001".

    Returns:
        (modalities, seg, voxel_volume_mm3) where modalities is a (4, D, H, W) array in T1/T1ce/T2/
        FLAIR order and voxel_volume_mm3 comes from the NIfTI header's spacing.
    """
    volumes = {}
    for mod in MODALITIES:
        path = os.path.join(patient_dir, f"{patient_id}_{mod}.nii")
        volumes[mod] = nib.load(path).get_fdata().astype(np.float32)
    seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii")
    if not os.path.exists(seg_path):
        # BraTS20_Training_355 ships with an unrenamed source filename instead of the standard pattern
        candidates = glob.glob(os.path.join(patient_dir, "*.nii"))
        candidates = [c for c in candidates if not any(f"_{m}.nii" in c for m in MODALITIES)]
        if len(candidates) != 1:
            raise FileNotFoundError(f"Could not uniquely identify seg file for {patient_id}: {candidates}")
        seg_path = candidates[0]
    seg_img = nib.load(seg_path)
    seg = seg_img.get_fdata().astype(np.uint8)
    voxel_spacing = seg_img.header.get_zooms()[:3]
    return volumes, seg, voxel_spacing


def zscore_normalize(volume, brain_mask):
    """Z-score normalize a volume using only voxels inside the brain mask.

    Normalizing over the whole array would be wrong for skull-stripped data: the large zero background
    would dominate the mean and standard deviation and compress the actual tissue intensities.

    Args:
        volume: single-modality 3D array.
        brain_mask: boolean mask of non-background voxels.

    Returns:
        Normalized array of the same shape, with background left at zero.
    """
    brain_voxels = volume[brain_mask]
    mean, std = brain_voxels.mean(), brain_voxels.std()
    normalized = np.zeros_like(volume)
    normalized[brain_mask] = (volume[brain_mask] - mean) / (std + 1e-8)
    return normalized


def resize_volume(volume, target_shape, order):
    """Resample a volume to the target grid.

    Args:
        volume: 3D array to resample.
        target_shape: desired output shape.
        order: spline interpolation order; 0 (nearest) for label masks so no new labels are invented,
            higher for continuous intensity data.

    Returns:
        Resampled array of shape target_shape.
    """
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)


def compute_subregion_volumes(seg, voxel_volume_mm3):
    """Compute true ET/TC/WT volumes in mm^3 from the native-resolution segmentation.

    Deliberately measured before resampling: the size bins that stratify every result in this project
    are derived from these numbers, so computing them on the 128^3 grid would fold resampling error
    into the independent variable.

    Args:
        seg: native-resolution integer label volume.
        voxel_volume_mm3: physical volume of one native voxel.

    Returns:
        dict mapping "ET"/"TC"/"WT" -> volume in mm^3.
    """
    et = int(np.sum(seg == 4))
    tc = int(np.sum((seg == 1) | (seg == 4)))
    wt = int(np.sum((seg == 1) | (seg == 2) | (seg == 4)))
    return {
        "et_voxels": et, "tc_voxels": tc, "wt_voxels": wt,
        "et_volume_mm3": et * voxel_volume_mm3,
        "tc_volume_mm3": tc * voxel_volume_mm3,
        "wt_volume_mm3": wt * voxel_volume_mm3,
    }


def process_patient(patient_dir, patient_id, target_size):
    """Preprocess one patient end to end and return its volumes for the summary table.

    Args:
        patient_dir: directory containing that patient's NIfTI files.
        patient_id: e.g. "BraTS20_Training_001".
        target_size: edge length of the isotropic output grid.

    Returns:
        dict of the patient's true per-region lesion volumes, for the summary CSV.
    """
    volumes, seg, spacing = load_patient(patient_dir, patient_id)
    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])

    brain_mask = np.zeros_like(seg, dtype=bool)
    for mod in MODALITIES:
        brain_mask |= volumes[mod] > 0

    normalized = [zscore_normalize(volumes[mod], brain_mask) for mod in MODALITIES]
    target_shape = (target_size, target_size, target_size)
    # order=1 (linear) for intensities, order=0 (nearest) for the mask so label values (0/1/2/4) survive resize
    image = np.stack([resize_volume(v, target_shape, order=1) for v in normalized], axis=0).astype(np.float32)
    mask_resized = resize_volume(seg.astype(np.float32), target_shape, order=0).astype(np.uint8)

    # lesion-volume stats are computed on the ORIGINAL segmentation/spacing, not the resized grid,
    # so size bins reflect true anatomical volume rather than resampling artifacts
    stats = compute_subregion_volumes(seg, voxel_volume_mm3)
    return image, mask_resized, stats


def main():
    """Preprocess every patient, writing results incrementally.

    Results are written per patient rather than once at the end: an earlier version accumulated
    everything in memory and wrote the summary CSV last, so the crash on patient 355's irregular
    filename would have silently discarded all preceding work.
    """
    parser = argparse.ArgumentParser(description="Preprocess BraTS2020 volumes.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N patients (for quick testing).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    patient_ids = sorted(
        p for p in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, p))
    )
    if args.limit:
        patient_ids = patient_ids[: args.limit]

    fieldnames = ["patient_id", "et_voxels", "tc_voxels", "wt_voxels", "et_volume_mm3", "tc_volume_mm3", "wt_volume_mm3"]
    csv_path = os.path.join(args.out_dir, "lesion_volumes.csv")
    done_ids = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            done_ids = {row["patient_id"] for row in csv.DictReader(f)}

    write_header = not os.path.exists(csv_path)
    n_processed = 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for i, patient_id in enumerate(patient_ids):
            if patient_id in done_ids:
                continue
            patient_dir = os.path.join(args.data_root, patient_id)
            image, mask, stats = process_patient(patient_dir, patient_id, args.target_size)
            np.savez_compressed(os.path.join(args.out_dir, f"{patient_id}.npz"), image=image, mask=mask)
            writer.writerow({"patient_id": patient_id, **stats})
            f.flush()
            n_processed += 1
            print(f"[{i + 1}/{len(patient_ids)}] {patient_id}: ET={stats['et_voxels']} TC={stats['tc_voxels']} WT={stats['wt_voxels']} voxels")

    print(f"\nDone. Processed {n_processed} new patients this run. Summary at {csv_path}")


if __name__ == "__main__":
    main()
