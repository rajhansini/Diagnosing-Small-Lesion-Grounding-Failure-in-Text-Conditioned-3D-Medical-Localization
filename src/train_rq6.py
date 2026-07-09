"""RQ6: attempt to fix the "large"-class embedding hub identified in RQ4's noise probe. Same
scale-matched patch training as RQ4, but adds a uniformity regularizer that directly penalizes
high pairwise cosine similarity among the projected text class embeddings, pushing them apart on
the hypersphere so no single class can act as a generic attractor regardless of input content."""
import argparse
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_rq4 import BraTSScaleMatchedPatchDataset
from model import TextVolumeAligner
from text_encoder import SIZE_CLASS_ORDER

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
LESION_CSV = os.path.join(PREPROCESSED_DIR, "lesion_volumes.csv")
TEXT_EMB_PATH = os.path.join(PREPROCESSED_DIR, "size_conditioned_text_embeddings.pt")
CKPT_DIR = "/net/projects/ranalab/rajhansini/nlp_project/checkpoints"


def get_patient_ids():
    return sorted(f[:-4] for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".npz"))


def uniformity_loss(txt_proj):
    """Mean pairwise cosine similarity among the (L2-normalized) projected text class embeddings.
    Minimizing this pushes classes apart, directly counteracting a single class acting as a
    generic attractor for any input, as RQ4's noise probe showed for the 'large' class."""
    sim_matrix = txt_proj @ txt_proj.T
    n = sim_matrix.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    return sim_matrix[off_diag].mean()


def run_epoch(model, loader, text_embeds, optimizer, temperature, uniformity_weight, device, train):
    model.train(train)
    total_ce, total_unif, correct, n = 0.0, 0.0, 0, 0
    for patches, labels in loader:
        patches, labels = patches.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            img_proj = model.encode_image(patches)
            txt_proj = model.encode_text(text_embeds)
            logits = img_proj @ txt_proj.T / temperature
            ce_loss = F.cross_entropy(logits, labels)
            unif_loss = uniformity_loss(txt_proj)
            loss = ce_loss + uniformity_weight * unif_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_ce += ce_loss.item() * len(labels)
        total_unif += unif_loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += len(labels)
    return total_ce / n, total_unif / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--uniformity_weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit_patients", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    random.seed(0)
    patient_ids = get_patient_ids()
    random.shuffle(patient_ids)
    if args.limit_patients:
        patient_ids = patient_ids[: args.limit_patients]

    n_val = max(1, int(0.2 * len(patient_ids)))
    val_ids, train_ids = patient_ids[:n_val], patient_ids[n_val:]

    train_ds = BraTSScaleMatchedPatchDataset(PREPROCESSED_DIR, LESION_CSV, train_ids)
    val_ds = BraTSScaleMatchedPatchDataset(PREPROCESSED_DIR, LESION_CSV, val_ids)
    print(f"train patients={len(train_ids)} patches={len(train_ds)} | val patients={len(val_ids)} patches={len(val_ds)}")
    print("classes:", SIZE_CLASS_ORDER, "| uniformity_weight:", args.uniformity_weight)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    text_embeds_dict = torch.load(TEXT_EMB_PATH, weights_only=True)
    text_embeds = torch.stack([text_embeds_dict[c] for c in SIZE_CLASS_ORDER]).to(args.device)

    model = TextVolumeAligner(text_dim=text_embeds.shape[1]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_val_acc, best_epoch = -1.0, -1
    for epoch in range(args.epochs):
        train_ce, train_unif, train_acc = run_epoch(model, train_loader, text_embeds, optimizer, args.temperature, args.uniformity_weight, args.device, train=True)
        val_ce, val_unif, val_acc = run_epoch(model, val_loader, text_embeds, optimizer, args.temperature, args.uniformity_weight, args.device, train=False)
        print(f"epoch {epoch + 1}/{args.epochs}: train_ce={train_ce:.4f} train_unif={train_unif:.4f} train_acc={train_acc:.3f} | val_ce={val_ce:.4f} val_unif={val_unif:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "rq6_uniformity_aligner_best.pt"))

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "rq6_uniformity_aligner_last.pt"))
    print(f"\nBest val_acc={best_val_acc:.3f} at epoch {best_epoch}. Saved best + last checkpoints to {CKPT_DIR}")


if __name__ == "__main__":
    main()
