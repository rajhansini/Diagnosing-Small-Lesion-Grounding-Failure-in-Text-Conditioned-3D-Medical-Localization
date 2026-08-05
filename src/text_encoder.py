"""PubMedBERT-based text encoder for subregion description embeddings."""
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

SUBREGION_DESCRIPTIONS = {
    "ET": [
        "Enhancing tumor: region of active contrast enhancement indicating viable, aggressive tumor tissue.",
        "The enhancing tumor is the area of the glioma that shows contrast uptake on T1-weighted post-contrast imaging.",
        "Enhancing tumor tissue represents the actively growing, vascularized portion of the brain tumor.",
    ],
    "TC": [
        "Tumor core: the necrotic and enhancing tumor mass, excluding surrounding edema.",
        "The tumor core comprises the enhancing tumor together with the necrotic and non-enhancing tumor tissue.",
        "Tumor core refers to the solid bulk of the glioma, distinct from the surrounding swollen brain tissue.",
    ],
    "WT": [
        "Whole tumor: the entire abnormal region including tumor core and peritumoral edema.",
        "The whole tumor extent includes the tumor core plus the surrounding edema visible on FLAIR imaging.",
        "Whole tumor describes the full abnormal signal region, encompassing both the tumor mass and swelling around it.",
    ],
    "NONE": [
        "Normal brain tissue: healthy tissue without tumor, edema, or necrosis.",
        "This region shows normal-appearing brain parenchyma with no evidence of tumor.",
        "Unremarkable brain tissue, free of any mass, enhancement, or surrounding edema.",
    ],
}

REGION_ORDER = ["ET", "TC", "WT", "NONE"]

# RQ5: naturalistic radiology-report-style phrasing (hedged, varied syntax, clinical narrative)
# instead of textbook-definition style, to test whether the size-collapse finding is an artifact
# of the templated language rather than a property of the model/task.
SUBREGION_DESCRIPTIONS_NATURALISTIC = {
    "ET": [
        "Post-contrast T1-weighted images demonstrate irregular nodular enhancement, favoring viable, high-grade tumor tissue.",
        "There is avid contrast enhancement within the mass, compatible with the actively proliferating component of the glioma.",
        "Following gadolinium administration, a focus of abnormal enhancement is identified, consistent with the tumor's biologically active portion.",
    ],
    "TC": [
        "The solid tumor mass, comprising both enhancing and necrotic components, is again demonstrated centrally within the lesion.",
        "Combining the necrotic and enhancing portions, the tumor core remains grossly unchanged in overall bulk.",
        "The compact, non-edematous tumor bulk -- necrotic center plus enhancing rim -- defines the surgical target volume.",
    ],
    "WT": [
        "The full extent of T2/FLAIR signal abnormality, encompassing the tumor and surrounding vasogenic edema, is again noted.",
        "Overall lesion burden, including infiltrative edema beyond the enhancing margin, appears similar to prior.",
        "The entire abnormal signal region -- tumor plus peritumoral edema -- extends into the adjacent white matter tracts.",
    ],
    "NONE": [
        "The remainder of the brain parenchyma is unremarkable, without abnormal signal, mass effect, or enhancement.",
        "No additional foci of enhancement or edema are identified elsewhere in the visualized brain.",
        "Background white and gray matter appear within normal limits for patient age.",
    ],
}


def embed_naturalistic_descriptions(tokenizer, model, device="cpu"):
    """Embed the RQ5 naturalistic, radiology-report-style region descriptions.

    These replace the templated textbook phrasing with hedged, syntactically varied language of the
    kind a radiologist would actually dictate, to test whether the project's findings are an artifact
    of template phrasing.

    Args:
        tokenizer: the PubMedBERT tokenizer.
        model: the frozen PubMedBERT encoder.
        device: torch device string.

    Returns:
        dict mapping region -> mean-pooled sentence embedding tensor.
    """
    embeddings = {}
    for region, texts in SUBREGION_DESCRIPTIONS_NATURALISTIC.items():
        emb = embed_texts(texts, tokenizer, model, device)
        embeddings[region] = emb.mean(dim=0)
    return embeddings

# RQ2: size-conditioned prompts, used as training targets (true size known at train time)
# and ensembled over at eval time (true size not known, so query all three and take the max response).
SIZE_CONDITIONED_DESCRIPTIONS = {
    ("ET", "small"): [
        "A small, subtle focus of enhancing tumor, only a few millimeters across.",
        "A tiny area of contrast-enhancing tumor tissue, easy to overlook.",
    ],
    ("ET", "medium"): [
        "A moderate-sized region of enhancing tumor.",
        "A medium area of active contrast enhancement within the glioma.",
    ],
    ("ET", "large"): [
        "A large, extensive region of enhancing tumor.",
        "A large, bulky area of contrast-enhancing tumor tissue.",
    ],
    ("TC", "small"): [
        "A small tumor core, a compact area of solid tumor mass.",
        "A small focus of necrotic and enhancing tumor tissue.",
    ],
    ("TC", "medium"): [
        "A moderate-sized tumor core.",
        "A medium-sized solid mass of necrotic and enhancing tumor.",
    ],
    ("TC", "large"): [
        "A large tumor core, a substantial mass of solid tumor tissue.",
        "A large, extensive area of necrotic and enhancing tumor.",
    ],
    ("WT", "small"): [
        "A small whole tumor extent, a limited area of abnormal signal.",
        "A small combined region of tumor and surrounding edema.",
    ],
    ("WT", "medium"): [
        "A moderate whole tumor extent, including a fair amount of surrounding edema.",
        "A medium-sized combined region of tumor core and edema.",
    ],
    ("WT", "large"): [
        "A large whole tumor extent, with extensive surrounding edema.",
        "A large, widespread combined region of tumor and swelling.",
    ],
}

SIZE_CONDITIONED_KEYS = list(SIZE_CONDITIONED_DESCRIPTIONS.keys())
# training/eval class order: ET_small, ET_medium, ET_large, TC_small, ..., WT_large, NONE
SIZE_CLASS_ORDER = [f"{r}_{s}" for r, s in SIZE_CONDITIONED_KEYS] + ["NONE"]


def embed_size_conditioned_descriptions(tokenizer, model, device="cpu"):
    """Returns dict: '{region}_{size_bin}' -> averaged embedding, plus 'NONE' from the base descriptions."""
    embeddings = {}
    for (region, size_bin), texts in SIZE_CONDITIONED_DESCRIPTIONS.items():
        emb = embed_texts(texts, tokenizer, model, device)
        embeddings[f"{region}_{size_bin}"] = emb.mean(dim=0)
    embeddings["NONE"] = embed_texts(SUBREGION_DESCRIPTIONS["NONE"], tokenizer, model, device).mean(dim=0)
    return embeddings


def load_text_encoder(device="cpu"):
    """Load the frozen PubMedBERT tokenizer and encoder.

    Args:
        device: torch device string to place the model on.

    Returns:
        (tokenizer, model) with the model in eval mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    return tokenizer, model


def embed_texts(texts, tokenizer, model, device="cpu"):
    """Mean-pool the last hidden state over non-padding tokens to get one embedding per input text."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state  # (batch, seq_len, dim)
    mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_subregion_descriptions(tokenizer, model, device="cpu"):
    """Returns dict: region -> averaged embedding across its template sentences."""
    embeddings = {}
    for region, texts in SUBREGION_DESCRIPTIONS.items():
        emb = embed_texts(texts, tokenizer, model, device)
        embeddings[region] = emb.mean(dim=0)
    return embeddings


if __name__ == "__main__":
    tokenizer, model = load_text_encoder()
    embeddings = embed_subregion_descriptions(tokenizer, model)
    for region, emb in embeddings.items():
        print(f"{region}: shape={tuple(emb.shape)}, norm={emb.norm().item():.3f}")

    print("\nPairwise cosine similarities (sanity check — should be well below 1.0, i.e. distinguishable):")
    regions = list(embeddings.keys())
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a, b = embeddings[regions[i]], embeddings[regions[j]]
            sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            print(f"  {regions[i]} vs {regions[j]}: {sim:.3f}")

    out_path = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed/subregion_text_embeddings.pt"
    torch.save({k: v.cpu() for k, v in embeddings.items()}, out_path)
    print(f"\nSaved embeddings to {out_path}")

    size_embeddings = embed_size_conditioned_descriptions(tokenizer, model)
    size_out_path = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed/size_conditioned_text_embeddings.pt"
    torch.save({k: v.cpu() for k, v in size_embeddings.items()}, size_out_path)
    print(f"Saved {len(size_embeddings)} size-conditioned embeddings to {size_out_path}")
    print("classes:", list(size_embeddings.keys()))

    naturalistic_embeddings = embed_naturalistic_descriptions(tokenizer, model)
    naturalistic_out_path = "/net/projects/ranalab/rajhansini/nlp_project/data/preprocessed/naturalistic_text_embeddings.pt"
    torch.save({k: v.cpu() for k, v in naturalistic_embeddings.items()}, naturalistic_out_path)
    print(f"\nSaved naturalistic-text embeddings to {naturalistic_out_path}")
    print("\nPairwise cosine similarities (naturalistic text):")
    nat_regions = list(naturalistic_embeddings.keys())
    for i in range(len(nat_regions)):
        for j in range(i + 1, len(nat_regions)):
            a, b = naturalistic_embeddings[nat_regions[i]], naturalistic_embeddings[nat_regions[j]]
            sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            print(f"  {nat_regions[i]} vs {nat_regions[j]}: {sim:.3f}")
