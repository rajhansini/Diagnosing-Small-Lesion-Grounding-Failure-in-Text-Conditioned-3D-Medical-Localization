"""P' reference point: conventional supervised BraTS segmentation on the identical split.

WHY THIS EXISTS. Every result in this project is produced by one pipeline -- our preprocessing, our
train/val split, our Dice implementation, our size-binning. If any of those carried a bug, the
headline "small lesions localize badly" finding would be unfalsifiable from the inside: we would have
no way to tell a real grounding failure apart from, say, a misaligned mask or a broken Dice. The
standard defence is to run the *same setup* on a previously-studied problem whose published numbers
are known, and check we land in the right range.

That problem here is supervised BraTS tumor segmentation, which the MICCAI BraTS challenge has scored
publicly since 2012. It reuses, unchanged:
  * the same preprocessed .npz volumes (preprocess.py)
  * the same seed-0 train/val split as train_baseline.py / evaluate_rq1.py
  * the same ET/TC/WT region definitions (dataset.region_mask)
  * the same dice_iou() and size_bin() used by every evaluation script
and changes only the model and the supervision signal: a conventional 3D U-Net trained with dense
voxel labels instead of a text-conditioned contrastive aligner. If this reaches Dice in the published
BraTS range, the shared machinery is sound and the text-conditioned results can be read as statements
about text conditioning rather than about our plumbing.

SECOND PURPOSE. This also answers a question the text-conditioned experiments cannot: does the
size-dependent collapse happen to *any* method on this data, or is it specific to text-conditioned
grounding? Because P' is scored by the identical size-stratified protocol, its large/small Dice ratio
is directly comparable to RQ1's, which separates "small lesions are intrinsically harder" from "text
conditioning fails on small lesions". See evaluate_pprime_supervised.py.

Trains with multi-label sigmoid Dice loss over the three nested BraTS regions (ET/TC/WT), which is
the formulation the challenge itself scores, not a 4-way mutually-exclusive labelling.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from torch.utils.data import DataLoader

from dataset_pprime import SEG_REGIONS, BraTSSegDataset
from evaluate_rq1 import dice_iou

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
CKPT_DIR = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints"


def get_patient_ids():
    """List preprocessed patient IDs in the fixed order every split in this project starts from.

    Must stay byte-identical to train_baseline.py's version: the sorted listing plus a seeded shuffle
    is what makes the P' split the same patients as RQ1's.
    """
    return sorted(f[:-4] for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz"))


def build_model(device):
    """Construct the 3D U-Net used as the supervised reference.

    Deliberately a plain, unremarkable configuration -- 4 downsampling stages, instance norm, 3
    sigmoid output channels. The point of P' is to show the *data and metric* pipeline is correct
    using an architecture whose expected performance is well established, not to compete on the BraTS
    leaderboard.

    Args:
        device: torch device string to place the model on.

    Returns:
        monai.networks.nets.UNet on the requested device.
    """
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=len(SEG_REGIONS),
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="instance",
    ).to(device)


def evaluate(model, loader, device, threshold=0.5):
    """Score the model on full volumes with the project's shared Dice implementation.

    Uses evaluate_rq1.dice_iou -- the same function that produced every Dice number in Sections 6-7 --
    so a discrepancy between P' and published BraTS numbers would implicate that function directly.

    Args:
        model: segmentation network in any mode; set to eval internally.
        loader: DataLoader over a full-volume (train=False) BraTSSegDataset.
        device: torch device string.
        threshold: sigmoid probability above which a voxel is predicted positive.

    Returns:
        dict mapping region name -> mean Dice over the loader.
    """
    model.eval()
    per_region = {r: [] for r in SEG_REGIONS}
    with torch.no_grad():
        for image, target in loader:
            image = image.to(device)
            probs = torch.sigmoid(model(image)).cpu().numpy()
            target = target.numpy()
            for b in range(probs.shape[0]):
                for ri, region in enumerate(SEG_REGIONS):
                    gt = target[b, ri] > 0.5
                    if not gt.any():
                        continue  # e.g. patients with no enhancing tumor; excluded exactly as in RQ1
                    dice, _ = dice_iou(probs[b, ri] > threshold, gt)
                    per_region[region].append(dice)
    return {r: float(np.mean(v)) if v else float("nan") for r, v in per_region.items()}


def main():
    """Train the P' supervised segmenter and checkpoint the best epoch by mean validation Dice."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop_size", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0,
                        help="must match train_baseline.py's seed so P' sees RQ1's exact split")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--max_hours", type=float, default=3.4,
                        help="stop cleanly before the cluster's 4h wall limit kills the job")
    parser.add_argument("--limit_patients", type=int, default=None, help="smoke tests only")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    random.seed(args.seed)
    patient_ids = get_patient_ids()
    random.shuffle(patient_ids)
    if args.limit_patients:
        patient_ids = patient_ids[: args.limit_patients]
    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids, train_ids = patient_ids[:n_val], patient_ids[n_val:]
    print(f"=== P' supervised reference | device={device} seed={args.seed} ===")
    print(f"train patients={len(train_ids)} | val patients={len(val_ids)} (identical split to RQ1)")

    train_ds = BraTSSegDataset(PREPROCESSED_DIR, train_ids, crop_size=args.crop_size, train=True)
    val_ds = BraTSSegDataset(PREPROCESSED_DIR, val_ids, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = build_model(device)
    loss_fn = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "pprime_unet_best.pt")
    best_mean = -1.0
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for image, target in train_loader:
            image, target = image.to(device), target.to(device)
            loss = loss_fn(model(image), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * image.shape[0]
            n += image.shape[0]
        scheduler.step()
        elapsed = (time.time() - started) / 3600.0
        msg = f"epoch {epoch + 1}/{args.epochs}: train_dice_loss={total / max(n, 1):.4f} ({elapsed:.2f}h)"

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            scores = evaluate(model, val_loader, device)
            mean_dice = float(np.mean([scores[r] for r in SEG_REGIONS]))
            msg += " | val " + " ".join(f"{r}={scores[r]:.4f}" for r in SEG_REGIONS)
            msg += f" mean={mean_dice:.4f}"
            if mean_dice > best_mean:
                best_mean = mean_dice
                torch.save(model.state_dict(), ckpt_path)
                msg += "  <- best, saved"
        print(msg, flush=True)

        if elapsed > args.max_hours:
            print(f"stopping at epoch {epoch + 1}: hit --max_hours={args.max_hours}")
            break

    print(f"\nBest mean validation Dice: {best_mean:.4f}")
    print(f"Saved best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
