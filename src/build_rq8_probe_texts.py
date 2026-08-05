"""RQ8: build the compositionality / negation probe text embeddings.

RQ7 asks whether swapping the text encoder changes anything. RQ8 asks the complementary and more
mechanistic question: does the trained model respond to *linguistic structure* at all, or is the
text query functioning as an opaque class identifier that happens to be spelled in English?

Five probe conditions per region, all embedded with the same frozen PubMedBERT and the same
mean-pooling used everywhere else in the project:

  original   the exact sentences used to train the baseline. Acts as a correctness check: its
             embedding must match subregion_text_embeddings.pt to floating-point tolerance, and
             its downstream Dice must reproduce RQ1's numbers exactly.
  negated    the same clinical content under explicit negation ("no enhancing tumor is seen").
             A model that understands negation should respond very differently; a bag-of-words
             matcher will barely move, because almost every content word is preserved.
  shuffled   the original sentences with word order destroyed by a fixed-seed permutation. Holds
             the bag of words exactly constant and removes only syntax.
  swapped    the original sentence frames with the region's head term replaced by a different
             region's term. Tests whether the response is driven by the specific anatomical
             referent or by generic tumor-ish vocabulary.
  generic    contentless filler naming no anatomy or pathology at all -- the floor condition.

Interpretation: if negated, shuffled and generic all produce heatmaps highly correlated with
original, the model is not using language compositionally and the project's "architecture, not
language" claim is supported mechanistically rather than merely by elimination.

Runs on CPU on the login node; the GPU job only torch.loads the .pt this writes.
"""
import os
import random

import torch

from text_encoder import (
    REGION_ORDER,
    SUBREGION_DESCRIPTIONS,
    embed_texts,
    load_text_encoder,
)

PREPROCESSED_DIR = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed"
BASELINE_EMB_PATH = os.path.join(PREPROCESSED_DIR, "subregion_text_embeddings.pt")
OUT_PATH = os.path.join(PREPROCESSED_DIR, "rq8_probe_text_embeddings.pt")
SHUFFLE_SEED = 0

NEGATED_DESCRIPTIONS = {
    "ET": [
        "Enhancing tumor is absent: there is no region of active contrast enhancement and no viable, aggressive tumor tissue.",
        "No enhancing tumor is present; the glioma shows no contrast uptake on T1-weighted post-contrast imaging.",
        "There is no enhancing tumor tissue and no actively growing, vascularized portion of the brain tumor.",
    ],
    "TC": [
        "Tumor core is absent: there is no necrotic and no enhancing tumor mass, excluding surrounding edema.",
        "No tumor core is present; there is neither enhancing tumor nor necrotic and non-enhancing tumor tissue.",
        "There is no tumor core and no solid bulk of the glioma, distinct from the surrounding swollen brain tissue.",
    ],
    "WT": [
        "Whole tumor is absent: there is no abnormal region, no tumor core and no peritumoral edema.",
        "No whole tumor extent is present; there is neither tumor core nor surrounding edema visible on FLAIR imaging.",
        "There is no whole tumor and no abnormal signal region, encompassing neither tumor mass nor swelling around it.",
    ],
    "NONE": [
        "This is not normal brain tissue: the tissue is not healthy and is not free of tumor, edema, or necrosis.",
        "This region does not show normal-appearing brain parenchyma and is not without evidence of tumor.",
        "Remarkable brain tissue, not free of mass, enhancement, or surrounding edema.",
    ],
}

# Same sentence frames as SUBREGION_DESCRIPTIONS, with the head anatomical term swapped for another
# region's term, so the vocabulary stays clinical but the referent changes.
SWAPPED_DESCRIPTIONS = {
    "ET": [
        "Peritumoral edema: region of surrounding fluid infiltration indicating swollen, non-tumor tissue.",
        "The peritumoral edema is the area of the glioma that shows fluid signal on T2-weighted imaging.",
        "Peritumoral edema tissue represents the swollen, infiltrated portion of the brain surrounding the tumor.",
    ],
    "TC": [
        "Peritumoral edema: the swollen non-tumor brain, excluding the solid tumor mass.",
        "The peritumoral edema comprises the infiltrated white matter together with surrounding fluid signal.",
        "Peritumoral edema refers to the swelling around the glioma, distinct from the solid tumor bulk.",
    ],
    "WT": [
        "Enhancing tumor: the small focal region of contrast uptake excluding edema and necrosis.",
        "The enhancing tumor extent includes only the contrast-avid rim, not the surrounding edema on FLAIR.",
        "Enhancing tumor describes the focal contrast-enhancing signal alone, excluding both mass and swelling.",
    ],
    "NONE": [
        "Enhancing tumor: region of active contrast enhancement indicating viable, aggressive tumor tissue.",
        "The enhancing tumor is the area of the glioma that shows contrast uptake on T1-weighted post-contrast imaging.",
        "Enhancing tumor tissue represents the actively growing, vascularized portion of the brain tumor.",
    ],
}

GENERIC_DESCRIPTIONS = [
    "A region of tissue within the imaged volume.",
    "This area of the scan is shown in the image.",
    "Some part of the acquired volume is depicted here.",
]


def shuffle_words(sentence, rng):
    """Destroy word order in one sentence while preserving its bag of words exactly.

    Args:
        sentence: the sentence to permute.
        rng: seeded random.Random, so the permutation is reproducible.

    Returns:
        The same words in a fixed pseudo-random order.
    """
    words = sentence.split()
    rng.shuffle(words)
    return " ".join(words)


def build_shuffled():
    """Build the word-order-destroyed variant of every region description.

    Returns:
        dict mapping region -> list of shuffled sentences.
    """
    rng = random.Random(SHUFFLE_SEED)
    return {region: [shuffle_words(s, rng) for s in texts]
            for region, texts in SUBREGION_DESCRIPTIONS.items()}


def main():
    """Embed all five probe conditions with the frozen text encoder and save them."""
    tokenizer, model = load_text_encoder()
    shuffled = build_shuffled()

    conditions = {
        "original": SUBREGION_DESCRIPTIONS,
        "negated": NEGATED_DESCRIPTIONS,
        "shuffled": shuffled,
        "swapped": SWAPPED_DESCRIPTIONS,
        "generic": {r: GENERIC_DESCRIPTIONS for r in REGION_ORDER},
    }

    out = {}
    for condition, per_region in conditions.items():
        for region in REGION_ORDER:
            out[f"{condition}|{region}"] = embed_texts(per_region[region], tokenizer, model).mean(dim=0).cpu()

    # Correctness gate: re-embedding the original sentences must reproduce the embeddings the
    # baseline was actually trained against. If this drifts, every downstream comparison is invalid.
    baseline = torch.load(BASELINE_EMB_PATH, weights_only=True)
    print("Verifying the 'original' condition reproduces subregion_text_embeddings.pt:")
    max_drift = 0.0
    for region in REGION_ORDER:
        drift = (out[f"original|{region}"] - baseline[region]).abs().max().item()
        max_drift = max(max_drift, drift)
        print(f"    {region}: max abs difference = {drift:.3e}")
    assert max_drift < 1e-4, f"original probe embeddings drifted from the trained baseline ({max_drift:.3e})"
    print(f"  OK (max drift {max_drift:.3e})")

    print("\nFrozen-PubMedBERT cosine similarity to the ORIGINAL query, per condition:")
    print("(this is the *pre-projection* geometry -- how much the text encoder itself distinguishes them)")
    header = f"{'condition':<12}" + "".join(f"{r:>10}" for r in REGION_ORDER)
    print(header)
    for condition in conditions:
        line = f"{condition:<12}"
        for region in REGION_ORDER:
            a = out[f"original|{region}"]
            b = out[f"{condition}|{region}"]
            sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            line += f"{sim:>10.4f}"
        print(line)

    print("\nExample shuffled sentences (syntax destroyed, bag of words preserved):")
    for region in ["ET"]:
        for original, shuf in zip(SUBREGION_DESCRIPTIONS[region], shuffled[region]):
            print(f"    original: {original}")
            print(f"    shuffled: {shuf}\n")

    torch.save(out, OUT_PATH)
    print(f"Saved {len(out)} probe embeddings ({len(conditions)} conditions x {len(REGION_ORDER)} regions) to {OUT_PATH}")


if __name__ == "__main__":
    main()
