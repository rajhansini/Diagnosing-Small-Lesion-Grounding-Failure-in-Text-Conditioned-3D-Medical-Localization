"""RQ7: build the text-side ablation ladder.

The project's headline claim is "architecture, not language." That claim is currently asserted
rather than measured: every experiment so far uses exactly one text encoder (PubMedBERT). This
script builds the control conditions needed to actually test how much the *language* side
contributes, by replacing the text embeddings while holding the volume encoder, the training
protocol, the split, and every hyperparameter fixed.

Four variants, forming a ladder from "real domain-specific language" down to "no language at all":

  pubmedbert    (already built by text_encoder.py -- the existing baseline, not rebuilt here)
  bertbase      general-domain BERT over the SAME sentences. Tests: does biomedical pretraining matter?
  randbert      BERT architecture, randomly initialized, frozen, over the SAME sentences.
                Tests: does pretraining of any kind matter, or just the transformer's inductive bias?
  randvec_ortho 4 random ORTHONORMAL vectors. No text, no encoder, no sentences. Best-case geometry.
                Tests: does the text encoder contribute anything beyond acting as a class-ID lookup?
  randvec_aniso 4 random vectors whose pairwise-similarity (Gram) structure MATCHES PubMedBERT's
                anisotropic ~0.99 geometry, but carry no semantic content.

Why the last variant exists: randvec_ortho is not a clean test on its own. Orthonormal vectors are
*better separated* than PubMedBERT's near-collinear embeddings (Section 5's anisotropy note), so if
randvec_ortho wins we cannot tell whether the cause is "semantics are useless" or "PubMedBERT's
geometry is bad." randvec_aniso holds the geometry fixed at PubMedBERT's own and strips only the
semantic content, separating those two explanations. Reporting randvec_ortho without randvec_aniso
would be a confound.

All variants are 768-d and norm-matched to PubMedBERT's per-class embedding norms, so the trainable
projection head sees inputs of comparable magnitude in every condition.

Runs on CPU on the login node; GPU jobs only ever torch.load the .pt files this writes.
"""
import os

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from text_encoder import REGION_ORDER, SUBREGION_DESCRIPTIONS, embed_texts

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
PUBMEDBERT_PATH = os.path.join(PREPROCESSED_DIR, "subregion_text_embeddings.pt")

BERTBASE_NAME = "bert-base-uncased"
SEED = 0


def pairwise_cosine_report(embeddings, label):
    """Print the pairwise cosine similarity structure -- the number that motivates this whole
    ablation (PubMedBERT sits at ~0.99, i.e. almost no discriminative signal before projection)."""
    regions = REGION_ORDER
    mat = torch.stack([embeddings[r] for r in regions])
    normed = torch.nn.functional.normalize(mat, dim=-1)
    sims = normed @ normed.T
    off_diag = sims[~torch.eye(len(regions), dtype=bool)]
    print(f"\n[{label}] pairwise cosine similarity (mean off-diagonal = {off_diag.mean():.4f})")
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            print(f"    {regions[i]} vs {regions[j]}: {sims[i, j]:+.4f}")
    return float(off_diag.mean())


def build_bertbase(target_norms):
    """General-domain BERT, same sentences, same mean-pooling as the PubMedBERT path."""
    tokenizer = AutoTokenizer.from_pretrained(BERTBASE_NAME)
    model = AutoModel.from_pretrained(BERTBASE_NAME).eval()
    embeddings = {}
    for region, texts in SUBREGION_DESCRIPTIONS.items():
        embeddings[region] = embed_texts(texts, tokenizer, model).mean(dim=0)
    return rescale_to(embeddings, target_norms)


def build_randbert(target_norms):
    """BERT architecture with randomly initialized weights, frozen. Isolates pretrained knowledge
    from the architecture's own inductive bias -- the sentences and pooling are identical."""
    tokenizer = AutoTokenizer.from_pretrained(BERTBASE_NAME)
    config = AutoConfig.from_pretrained(BERTBASE_NAME)
    torch.manual_seed(SEED)
    model = AutoModel.from_config(config).eval()
    embeddings = {}
    for region, texts in SUBREGION_DESCRIPTIONS.items():
        embeddings[region] = embed_texts(texts, tokenizer, model).mean(dim=0)
    return rescale_to(embeddings, target_norms)


def random_orthonormal_rows(n_rows, dim, generator):
    """n_rows orthonormal vectors in R^dim, drawn uniformly (QR of a Gaussian matrix)."""
    gaussian = torch.randn(dim, n_rows, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(gaussian)  # (dim, n_rows), columns orthonormal
    return q.T  # (n_rows, dim), rows orthonormal


def build_randvec_ortho(target_norms):
    """No text at all: 4 orthonormal random directions, one per class. Maximally separated geometry."""
    generator = torch.Generator().manual_seed(SEED)
    rows = random_orthonormal_rows(len(REGION_ORDER), 768, generator).float()
    return rescale_to({r: rows[i] for i, r in enumerate(REGION_ORDER)}, target_norms)


def build_randvec_aniso(pubmedbert, target_norms):
    """No text, but PubMedBERT's *geometry* preserved exactly.

    Take G = the Gram matrix of PubMedBERT's L2-normalized class embeddings (so G holds the
    pairwise cosine similarities, ~0.99 off-diagonal). Factor G = L L^T via eigendecomposition
    (Cholesky is unsafe here: G is near rank-deficient precisely because the embeddings are nearly
    collinear). Then A = L Q, for Q a random orthonormal basis with rows in R^768, satisfies
    A A^T = L Q Q^T L^T = L L^T = G. So A's rows reproduce PubMedBERT's similarity structure while
    pointing in random directions carrying no semantic content.
    """
    mat = torch.stack([pubmedbert[r] for r in REGION_ORDER]).double()
    normed = torch.nn.functional.normalize(mat, dim=-1)
    gram = normed @ normed.T

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp(min=0.0)  # numerical negatives on a PSD matrix
    factor = eigenvectors @ torch.diag(eigenvalues.sqrt())  # L, with L L^T = G

    generator = torch.Generator().manual_seed(SEED + 1)
    basis = random_orthonormal_rows(len(REGION_ORDER), 768, generator)  # Q
    rows = (factor @ basis).float()

    embeddings = {r: rows[i] for i, r in enumerate(REGION_ORDER)}
    achieved = torch.nn.functional.normalize(rows, dim=-1) @ torch.nn.functional.normalize(rows, dim=-1).T
    max_error = (achieved.double() - gram).abs().max().item()
    print(f"\n[randvec_aniso] max |achieved - target| cosine error: {max_error:.2e} (should be ~1e-6)")
    return rescale_to(embeddings, target_norms)


def build_whitened(pubmedbert, target_norms):
    """PubMedBERT's real embeddings, with the anisotropy removed and the semantics kept.

    RQ7 established that random vectors carrying PubMedBERT's anisotropic geometry perform as well
    as PubMedBERT itself, while orthonormal random vectors are reliably worse. That points at the
    geometry rather than the meaning, but it cannot finish the argument, because every condition
    that had the geometry also lacked semantics and vice versa. The missing cell is the one that
    keeps the meaning and discards only the geometry.

    Whitening supplies it. Let M be the 4 x 768 matrix of L2-normalized class embeddings and
    G = M M^T their Gram matrix, whose off-diagonal entries are the ~0.99 similarities. Applying
    W = G^(-1/2) on the left gives M' = W M with Gram W G W^T = I: the four vectors are now mutually
    orthogonal, so the anisotropy is gone. Crucially W is invertible, so M' spans exactly the same
    subspace as M and each row remains a linear combination of the original embeddings -- no
    semantic content has been added or destroyed, only the angles between the classes.

    The prediction this tests: if RQ7's geometry story is the whole story, whitened embeddings
    should perform like randvec_ortho, which shares their geometry and has no semantics at all. If
    they instead perform like PubMedBERT, the meaning was doing work that RQ7's design could not see.
    """
    mat = torch.stack([pubmedbert[r] for r in REGION_ORDER]).double()
    normed = torch.nn.functional.normalize(mat, dim=-1)
    gram = normed @ normed.T

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp(min=1e-10)   # G is near rank-deficient by construction
    whitener = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T   # G^(-1/2)
    rows = (whitener @ normed).float()

    embeddings = {r: rows[i] for i, r in enumerate(REGION_ORDER)}
    achieved = torch.nn.functional.normalize(rows, dim=-1) @ torch.nn.functional.normalize(rows, dim=-1).T
    off = achieved[~torch.eye(len(REGION_ORDER), dtype=bool)]
    print(f"\n[whitened] max |off-diagonal cosine| after whitening: {off.abs().max().item():.2e} "
          f"(should be ~0; PubMedBERT's is ~0.99)")
    return rescale_to(embeddings, target_norms)


def rescale_to(embeddings, target_norms):
    """Match each class embedding's L2 norm to PubMedBERT's, so the trainable projection head sees
    comparable input magnitudes across variants and scale is not a confound."""
    return {r: v / (v.norm() + 1e-12) * target_norms[r] for r, v in embeddings.items()}


def main():
    """Build and save text embeddings for every RQ7 encoder-ablation condition."""
    pubmedbert = torch.load(PUBMEDBERT_PATH, weights_only=True)
    target_norms = {r: pubmedbert[r].norm().item() for r in REGION_ORDER}
    print("PubMedBERT per-class embedding norms (all variants are matched to these):")
    for r in REGION_ORDER:
        print(f"    {r}: {target_norms[r]:.3f}")
    baseline_aniso = pairwise_cosine_report(pubmedbert, "pubmedbert (existing baseline)")

    variants = {
        "bertbase": build_bertbase(target_norms),
        "randbert": build_randbert(target_norms),
        "randvec_ortho": build_randvec_ortho(target_norms),
        "randvec_aniso": build_randvec_aniso(pubmedbert, target_norms),
        "whitened": build_whitened(pubmedbert, target_norms),
    }

    print("\n" + "=" * 78)
    summary = {"pubmedbert": baseline_aniso}
    for name, embeddings in variants.items():
        summary[name] = pairwise_cosine_report(embeddings, name)
        out_path = os.path.join(PREPROCESSED_DIR, f"rq7_{name}_text_embeddings.pt")
        torch.save({k: v.cpu() for k, v in embeddings.items()}, out_path)
        print(f"    -> saved {out_path}")

    print("\n" + "=" * 78)
    print("Mean off-diagonal cosine similarity by variant (the geometry each model must work with):")
    for name, value in summary.items():
        print(f"    {name:<16} {value:+.4f}")
    print("\nSanity expectations: pubmedbert ~ +0.99 (anisotropic), randvec_ortho = 0.0000 (orthonormal),")
    print("randvec_aniso should match pubmedbert almost exactly, isolating semantics from geometry.")


if __name__ == "__main__":
    main()
