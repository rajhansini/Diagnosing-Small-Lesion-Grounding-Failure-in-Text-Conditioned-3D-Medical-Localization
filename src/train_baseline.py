"""Alignment baseline (RQ1): train image/text projections via 4-way (ET/TC/WT/NONE) contrastive
classification on 3D patches. Every text-conditioned result in the report builds on this model.

Naming note: earlier drafts called this "P'". That label now refers to the supervised segmentation
reference in train_pprime_supervised.py -- the previously-studied problem used to validate that the
shared data/split/metric pipeline is sound, which is what P' conventionally denotes."""
import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import BraTSPatchDataset
from model import TextVolumeAligner
from text_encoder import REGION_ORDER

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
TEXT_EMB_PATH = os.path.join(PREPROCESSED_DIR, "subregion_text_embeddings.pt")
CKPT_DIR = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints"


def get_patient_ids():
    """List preprocessed patient IDs in the fixed order every split starts from.

    Sorting before the seeded shuffle is what makes a given seed reproduce the same train/val split
    across every script in the project.

    Returns:
        Sorted list of patient ID strings.
    """
    return sorted(f[:-4] for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz"))


def run_epoch(model, loader, text_embeds, optimizer, temperature, device, train):
    """Run one training or evaluation epoch of contrastive classification.

    Image and text embeddings are projected into the shared space, cosine similarities are scaled by
    1/temperature to form logits, and cross-entropy is taken against the true class. This is the
    InfoNCE-style contrastive objective with the text side acting as a fixed set of class prototypes.

    Args:
        model: TextVolumeAligner being trained or evaluated.
        loader: DataLoader yielding (patch, label) batches.
        text_embeds: (n_classes, text_dim) frozen sentence embeddings for every class.
        optimizer: torch optimizer; unused when train is False.
        temperature: softmax temperature applied to the cosine-similarity logits.
        device: torch device string.
        train: True to backpropagate and step, False for a no-grad evaluation pass.

    Returns:
        (mean loss, accuracy) over the epoch.
    """
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for patches, labels in loader:
        patches, labels = patches.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            img_proj = model.encode_image(patches)
            txt_proj = model.encode_text(text_embeds)
            logits = img_proj @ txt_proj.T / temperature
            loss = F.cross_entropy(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


def main():
    """Train the P' contrastive baseline and save its checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit_patients", type=int, default=None, help="for quick smoke tests")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="controls the train/val split, for cross-validation across seeds")
    args = parser.parse_args()

    random.seed(args.seed)
    patient_ids = get_patient_ids()
    random.shuffle(patient_ids)
    if args.limit_patients:
        patient_ids = patient_ids[: args.limit_patients]

    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids, train_ids = patient_ids[:n_val], patient_ids[n_val:]

    train_ds = BraTSPatchDataset(PREPROCESSED_DIR, train_ids, patch_size=args.patch_size)
    val_ds = BraTSPatchDataset(PREPROCESSED_DIR, val_ids, patch_size=args.patch_size)
    print(f"train patients={len(train_ids)} patches={len(train_ds)} | val patients={len(val_ids)} patches={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    text_embeds_dict = torch.load(TEXT_EMB_PATH)
    text_embeds = torch.stack([text_embeds_dict[r] for r in REGION_ORDER]).to(args.device)

    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(CKPT_DIR, exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, text_embeds, optimizer, args.temperature, args.device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, text_embeds, optimizer, args.temperature, args.device, train=False)
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    ckpt_name = "baseline_aligner.pt" if args.seed == 0 else f"baseline_aligner_seed{args.seed}.pt"
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {ckpt_path}")


if __name__ == "__main__":
    main()
