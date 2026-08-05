---
title: "Diagnosing Small-Lesion Grounding Failure in Text-Conditioned 3D Medical Localization"
author: "MPCS 53113 Natural Language Processing — Project Check-in"
date: ""
geometry: margin=0.45in
fontsize: 8.5pt
mainfont: "DejaVu Serif"
header-includes:
  - \usepackage{multicol}
  - \usepackage{enumitem}
  - \setlength{\columnsep}{20pt}
  - \setlist{itemsep=1pt,parsep=0pt,topsep=1pt,partopsep=0pt,leftmargin=1.1em}
  - \pagestyle{empty}
  - \setlength{\parskip}{2pt}
---

## Idea

Text-conditioned 3D medical vision-language models can often detect *that* a lesion exists but fail to localize it precisely when it is small, defaulting to imprecise, oversized regions. This is noted qualitatively in recent 3D medical VLM literature but not rigorously *quantified* as a function of lesion size on volumetric data. This project (1) quantifies that failure on BraTS2020 brain MRI, then (2) runs a series of ablations against candidate fixes, with statistically corrected paired testing throughout.

## Plan

- Build a contrastive text-volume alignment model: PubMedBERT sentence embeddings (mean-pooled) aligned with 3D ResNet-10 (MONAI) patch embeddings, on BraTS2020, 4 classes (enhancing tumor / tumor core / whole tumor / normal tissue).
- Measure localization (Dice/IoU via sliding-window similarity + Otsu thresholding — Grad-CAM isn't viable on a globally-pooled architecture) stratified by small/medium/large lesion-size terciles, vs. a chance-level random-heatmap control.
- Ablate candidate fixes: size-conditioned text prompts (RQ2), multi-scale/smaller sliding windows (RQ3/3b/3c), scale-matched retraining (RQ4), an embedding-collapse repair (RQ6), naturalistic report-style text (RQ5).
- Validate every claim with paired Wilcoxon tests + Benjamini-Hochberg FDR correction across the full accumulated family of tests (96 by the end).

## What's been done

- **Baseline (P′) validated**: 4-way val. accuracy 0.671 vs. 0.25 chance — genuine signal learned.
- **RQ1 — core finding**: severe, monotonic size-dependent collapse (5–15× Dice degradation, large→small) across all 3 tumor subregions; survives a chance-level control; replicates exactly (9/9 region×bin) on 2 additional independent splits.
- **RQ2 — size-conditioned prompting**: helps medium/large lesions, but *significantly worsens* small enhancing-tumor localization (p=0.010) — the opposite of its goal. Points to an architectural (fixed receptive field), not linguistic, bottleneck.
- **RQ3/3b/3c — window size**: naive multi-scale ensembling mostly hurts; isolating one smaller window helps robustly (TC/WT clear correction at 16³; all regions clear at 8³); trend plateaus at 6³ for ET/TC but whole tumor keeps improving.
- **RQ4 — scale-matched retraining**: better classification (0.668) but *significantly worse* localization everywhere; a noise-only probe (ANOVA p≈4×10⁻²⁵¹) proves the model exploits resize-interpolation artifacts, not real content — a directly-verified shortcut, with a generic bias toward the "large" class acting as an embedding hub.
- **RQ6 — hub repair**: a uniformity regularizer separates the embeddings, fixes 2/3 shortcut behaviors, beats RQ4 in all 9 bins — but still falls short of the plain baseline in most bins.
- **RQ5 — naturalistic text**: reproduces the original finding almost exactly — rules out template phrasing as a confound.
- **Bottom line**: the simplest intervention (evaluate the original model at one smaller window, no retraining) beats every more sophisticated fix attempted.

## Challenges

- **Environment**: a `cu12`/`cu13` NVIDIA package collision silently deleted working CUDA libraries after a MONAI install pulled in a newer PyTorch; separately, PubMedBERT's checkpoint format needed an older `transformers` pin to avoid re-triggering that conflict. Also a Kaggle API/Python-version mismatch, and no GPU on the login node (had to learn the cluster's SLURM partitions/QOS from scratch).
- **Data**: one BraTS patient ships a non-standard segmentation filename (a hospital de-identification leftover), which crashed preprocessing partway through and exposed a bug where results weren't saved incrementally — fixed both.
- **Design**: Grad-CAM was the original localization plan but was caught as a dead end *before* implementation — the architecture globally pools each patch, leaving no spatial map to back-propagate onto.
- **Methodology**: raw PubMedBERT embeddings for the 4 classes have ~0.99 pairwise cosine similarity (BERT anisotropy) — the trainable projection head has to do all the discriminative work.
- **Statistics**: with 96 accumulated tests, several ablation "wins" (e.g. RQ3b's ET-region improvement) do *not* survive Benjamini-Hochberg correction — required reporting some effects as real-but-not-significant rather than rounding up.

## Where we'd like feedback

- All results are on one anatomy/dataset (BraTS brain tumors) — is that an acceptable scope for the final report, or should generalization to a second organ/dataset be expected?
- RQ6's embedding-hub fix is partial (2/3 shortcut behaviors fixed, still below baseline) — worth continuing to chase toward full repair, or is "simplest fix wins, and here's honestly why the fancier ones don't" a satisfying enough final story?
- Per-bin sample sizes are modest (n=20–32 patients) — is paired Wilcoxon + BH-FDR sufficient rigor here, or would a full k-fold sweep across RQ2–RQ6 (not just RQ1) be expected?
