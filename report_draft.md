# Architecture, Not Language: Diagnosing Small-Lesion Grounding Failure in Text-Conditioned 3D Medical Localization

**MPCS 53113 Natural Language Processing — Final Report**
University of Chicago

Source code: [https://github.com/rajhansini/Diagnosing-Small-Lesion-Grounding-Failure-in-Text-Conditioned-3D-Medical-Localization](https://github.com/rajhansini/Diagnosing-Small-Lesion-Grounding-Failure-in-Text-Conditioned-3D-Medical-Localization)

## Abstract

Recent 3D medical vision-language models can detect *that* a pathological finding is present but often fail to say *where* when the finding is small, defaulting to oversized, imprecise regions. This has been reported qualitatively but not quantified as a function of lesion size on volumetric data. We build a text-conditioned contrastive localization pipeline — PubMedBERT sentence embeddings aligned with 3D ResNet-10 patch embeddings on BraTS2020 brain MRI — and measure Dice stratified by true lesion volume across three tumor subregions, running fifteen controlled experiments and 171 FDR-corrected paired tests.

**The failure is real and specific.** Localization collapses monotonically with lesion size (5–15× Dice degradation, large to small), survives a chance-level control, and replicates in 9 of 9 region×bin comparisons across three independent splits. To rule out the deflationary reading that small lesions are simply hard here, we train a conventional supervised U-Net on the identical data, split and metric: it reaches published BraTS Dice (ET 0.758, TC 0.812, WT 0.851) and degrades by only 1.2–1.3× large-to-small. Small lesions are learnable on this data; text-conditioned grounding is what fails on them.

**Roughly half the measured collapse was a metric artifact.** Decomposing the pipeline shows the unsupervised Otsu threshold over-predicts lesion volume by 6–223× and is *anti*-correlated with true size (ρ = −0.26 to −0.48), mechanically inflating the effect. Under an oracle-volume threshold the large/small ratio falls from 15.0/9.2/4.9 to 14.4/5.6/2.4. A threshold-free pointing game localizes what remains: chance-corrected accuracy is roughly uniform at 46–122× chance everywhere, with one stark exception — for small enhancing tumor the peak response lands inside the lesion in **0 of 21 patients**, exactly chance.

**The language pathway contributes almost nothing.** Replacing PubMedBERT with general-domain BERT, with a randomly initialized never-trained BERT, or with random vectors carrying no language at all produces effects that change sign across training runs — indistinguishable from the 0.0044 Dice noise floor of simply retraining the baseline. Only discarding PubMedBERT's anisotropic embedding *geometry* reliably hurts (−0.0087, consistent across three seeds). Probing the trained model directly, negating the query, destroying its word order, swapping its anatomical referent, or replacing it with contentless filler all leave the projected embedding above 0.94 cosine to the original and the heatmap at ρ = 0.56–1.00; for whole tumor, generic filler and a wrong-region term both *outperform* the true clinical description. The query functions as an opaque class identifier that happens to be spelled in English.

**One intervention works, and it is the simplest.** Size-conditioned prompting (whose reported gain turns out to be an artifact of the binarizer, vanishing under every calibrated rule), multi-scale ensembling, scale-matched retraining (which a noise probe shows learned resize-interpolation shortcuts, ANOVA p≈4×10⁻²⁵¹) and a uniformity regularizer all fail to beat the plain baseline. What does work is evaluating the frozen model at a smaller query window, around 12³–16³ instead of 32³, with no retraining: +0.060 Dice under a deployable top-1% threshold and +0.067 under an oracle — 2–3× more than the Otsu protocol that discovered it could show. It also lifts threshold-free pointing accuracy for enhancing tumor in all three size bins, taking the previously-at-chance small-ET bin from 0 of 21 patients to 4 of 21.

**The methodological result is the transferable one.** Five conclusions here were overturned or materially qualified *after* being written up, by controls rather than by significance tests — including twice within one section, in opposite directions, once a confound between window size and tiling convention was removed. The general lesson: a shared confound is not a cancelled confound. Every comparison in this report used one binarizer, which we argued made them internally fair; Otsu and every calibrated rule turn out to disagree in *sign* about the same pair of heatmaps, so arms that interact differently with a shared instrument can be ranked backwards by it, invisibly to any amount of paired testing or FDR correction.

## 1. Introduction

Radiology reports routinely describe findings in natural language — location, size, character — while the underlying evidence lives in a 3D volume (CT/MRI). A model that could ground free text directly onto the matching 3D region would be useful for report-to-image verification, weakly-supervised segmentation without costly voxel-level labels, and explainable AI-assisted diagnosis. Several recent 3D medical vision-language models attempt exactly this, and a consistent, troubling pattern has emerged in the literature: these models can often tell *that* a finding exists, but when the finding is small, they fail to say *where* — collapsing to an imprecise region spanning a whole organ or quadrant rather than the actual lesion.

This is not a cosmetic failure. Small, subtle findings are disproportionately the clinically important case: large, obvious masses rarely need AI assistance to be found, while small or early-stage lesions are exactly where a grounding tool would be most valuable — and exactly where current systems are documented to fail worst. If this failure mode is real but unquantified, it's difficult to know how bad it is, whether it's fixable, or what a fix would even target.

This project does four things. First, it builds a controlled experimental setup to **rigorously quantify** this failure as a function of lesion size, rather than relying on qualitative or anecdotal reports — measuring localization quality (Dice/IoU) stratified into small/medium/large tercile bins, on a held-out validation set, across three independently-defined tumor subregions in the BraTS2020 dataset. Second, it **validates that setup against a previously-studied problem** (P′): a conventional supervised segmenter trained on the identical data, split and metric, which both confirms the pipeline is sound and establishes that the collapse is specific to text conditioning rather than intrinsic to small lesions on this data. Third, it runs a systematic **series of ablation studies** — text-side, inference-side, and training-side interventions — to see whether the failure is a *language* problem, an *architecture* problem, or something that can be trained away. Fourth, throughout, it applies a consistent set of **diagnostic controls** (Section 8: chance baselines, shortcut-learning noise probes, cross-split and cross-run replication, family-wise multiple-comparisons correction, pipeline decomposition, metric triangulation, metric well-definedness, and confound isolation) so that every claim of "improvement" or "failure" is verified rather than eyeballed.

Two things about how this turned out are worth stating up front. The failure is **architectural rather than linguistic** — strikingly so: the text query turns out to function as an opaque class identifier, and replacing the language model with random vectors changes nothing measurable. And exactly one intervention works, the simplest one available: querying the frozen model at a smaller window. That conclusion was reached only after the diagnostic controls in Section 8 overturned four earlier conclusions this report had already written down, including reversing the window-size verdict twice. Those reversals are documented rather than hidden, because how they were caught is the most transferable part of the work.

## 2. Problem Definition

Given a 3D medical volume $V$ and a text description $t$ of a finding, a text-conditioned localization model $f(V, t) \to M$ predicts a spatial region $M$ corresponding to that finding. Let $M^*$ be the ground-truth region with voxel volume $|M^*|$. We study the relationship between localization quality (Dice($M$, $M^*$)) and $|M^*|$: does quality degrade smoothly, or collapse below some size threshold? This has been noted as a real limitation in recent 3D medical VLM literature but not rigorously measured in a controlled, size-stratified way on volumetric data — that is the gap this project addresses.

## 3. Related Work

**Contrastive vision-language pretraining.** CLIP (Radford et al., 2021) established the pattern this project builds on: align image and text embeddings in a shared space via contrastive learning, then use the aligned space for zero-shot classification, retrieval, or (with post-hoc techniques) localization. We adapt this pattern to 3D volumetric patches rather than 2D natural images.

**BERT sentence embeddings and anisotropy.** Reimers & Gurevych's Sentence-BERT (2019) showed that raw BERT embeddings (CLS-token or mean-pooled) cluster tightly in a narrow cone of the embedding space and perform poorly on semantic similarity tasks without task-specific fine-tuning — a property called anisotropy. We observed exactly this with PubMedBERT (Section 5): our four class descriptions have ~0.99 pairwise cosine similarity in the raw embedding space, confirming the phenomenon holds for biomedical BERT variants too, not just general-domain BERT.

**Text-driven medical segmentation.** MedCLIP-SAMv2 and SimTxtSeg are recent frameworks that use text prompts to drive weakly-supervised or zero-shot medical image segmentation, integrating CLIP-style models with SAM-style segmentation. These operate primarily on 2D slices or 2D imaging modalities (e.g. chest X-ray, 2D CT slices); this project instead works with genuinely volumetric 3D patches and a sliding-window localization mechanism suited to that setting.

**3D medical vision-language grounding failures.** Several 2025-2026 papers document the specific failure this project quantifies. A benchmark of 3D medical VQA models found that "without explicit spatial localization, VLMs fail to attend to subtle lesion signals in raw 3D volumes... for small targets such as lesions or nodules, models default to large bounding boxes encompassing the entire organ or image quadrant." An audit of frontier medical VLMs similarly found grounding to be "a major failure point," with small-lesion measurement specifically flagged as harder than existence detection due to limited annotated small-lesion cases. These papers establish that the failure is real and current, but report it qualitatively or as one metric among many in a broader benchmark — none isolate the *relationship between lesion size and localization quality* with a controlled chance-level comparison, which is the specific gap this project fills. That comparison matters because Dice is known to be geometrically harsher on small structures for *any* method (Section 6) — without a chance baseline, a raw "small lesions score worse" result is not yet evidence of a model-specific failure, just a property of the metric. This project's contribution is showing the failure survives that control.

**BraTS.** The BraTS challenge (Menze et al., 2015; continued annually since) is the standard benchmark for brain tumor segmentation from multi-modal MRI, and is the data source for this project (2020 edition).

## 4. Dataset

**BraTS2020** (Kaggle: `awsaf49/brats20-dataset-training-validation`), sourced from the MICCAI Brain Tumor Segmentation Challenge. 369 glioma patients, each with 4 co-registered, skull-stripped MRI sequences (T1, T1ce, T2, FLAIR) and a voxel-level expert segmentation mask with three standard evaluation regions:
- **ET** (enhancing tumor)
- **TC** (tumor core = ET ∪ necrotic core)
- **WT** (whole tumor = TC ∪ peritumoral edema)

One patient (`BraTS20_Training_355`) ships with a non-standard segmentation filename (`W39_1998.09.19_Segm.nii`), a leftover artifact from the original hospital anonymization process never corrected in this release — handled with a filename fallback in preprocessing (see Challenges).

All volumes were intensity-normalized (z-score within the brain mask, per modality) and resampled to 128³. True lesion volumes (mm³) were computed from the native-resolution segmentation and voxel spacing (not the resampled grid), then used to bin patients into small/medium/large terciles per region:

| Region | Small cutoff | Large cutoff | Patients with zero volume |
|---|---|---|---|
| ET | ≤9,051 mm³ | >26,245 mm³ | 27/369 (no enhancing component) |
| TC | ≤19,191 mm³ | >49,948 mm³ | 0/369 |
| WT | ≤63,126 mm³ | >125,309 mm³ | 0/369 |

Split: 296 train / 73 held-out validation patients (80/20, fixed seed).

## 5. Method

**Text encoder**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`, mean-pooled over non-padding tokens. Four base region descriptions (ET/TC/WT/NONE, ~3 template sentences each). Raw PubMedBERT sentence embeddings for these four classes have ~0.99 pairwise cosine similarity — BERT anisotropy (Section 3), not a bug. This means the *trainable projection head*, not the frozen text encoder, is responsible for introducing discriminability.

**Volume encoder**: MONAI 3D ResNet-10, 4 input channels, operating on 32³-voxel patches sampled from the 128³ volume (positive patches centered on a random voxel within the target region's mask; a background/"NONE" class sampled from outside the whole-tumor region).

**Alignment (the *alignment baseline*)**: contrastive classification — image and text embeddings are projected into a shared 256-d space (L2-normalized), and trained with cross-entropy over cosine-similarity logits (temperature 0.07) against the 4 classes (ET/TC/WT/NONE). This is the model every text-conditioned result in this report is built on. Note on naming: earlier drafts called this "P′". That label is now reserved for the *previously-studied problem* used to validate the pipeline (supervised BraTS segmentation, Section 6.1), which is what P′ conventionally means; this contrastive model is referred to throughout as the alignment baseline or the RQ1 baseline.

**Localization / heatmap extraction**: Grad-CAM was the original plan but was rejected on inspection (Section 8) — this architecture globally average-pools each 32³ patch to a single embedding, leaving no meaningful spatial feature map near the output to back-propagate onto. Instead, we use a **sliding-window similarity map**: the trained patch encoder is swept across the full 128³ volume (stride 16), and each window's cosine similarity to the query text embedding is accumulated into a per-voxel heatmap. Validated with a sanity check confirming the ET-query heatmap scores higher inside the true ET region than outside (+0.247 mean difference) and the inverse holds for the NONE query.

**Binarization**: Otsu's method (unsupervised, per-volume) converts the continuous heatmap into a predicted mask for Dice/IoU scoring — chosen so the threshold is not tuned against ground truth (which would leak test-time information).

![Pipeline schematic: contrastive text-volume alignment at training time (left/center), and sliding-window heatmap extraction at inference time (right).](figures/fig_architecture.png){width=95%}

### 5.1 Techniques used, and how they work

Several methods this project relies on were not covered in class. Each is summarized here, with a reference to a fuller description.

**Contrastive alignment with a temperature-scaled softmax (InfoNCE).** Image and text embeddings are L2-normalized and projected into a shared space; the cosine similarity between an image embedding and each of the $K$ class text embeddings forms a logit vector, divided by a temperature $\tau$ (here 0.07) and passed through softmax cross-entropy against the true class:
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(v, t_{y})/\tau)}{\sum_{k=1}^{K}\exp(\text{sim}(v, t_{k})/\tau)}$$
Low $\tau$ sharpens the distribution, penalizing near-misses heavily and pushing classes apart. This is the CLIP/InfoNCE objective (Oord et al., 2018; Radford et al., 2021) with the text side acting as a fixed set of class prototypes rather than as in-batch negatives.

**Otsu's method** (Otsu, 1979). An unsupervised way to pick a binarization threshold from a single image's intensity histogram. It considers every candidate cut point and selects the one maximizing the variance *between* the two resulting groups — equivalently minimizing variance within them. We chose it precisely because it never sees ground truth, so it cannot leak test-time information into Dice. Section 6.3 quantifies what that choice cost.

**The uniformity regularizer** (Wang & Isola, 2020). Wang and Isola showed contrastive representation quality decomposes into *alignment* (matched pairs are close) and *uniformity* (embeddings spread over the hypersphere rather than collapsing). RQ4 produced exactly a uniformity failure — the class embeddings collapsed into a hub. RQ6 adds a penalty on high pairwise cosine similarity among projected class embeddings to push them apart.

**The pointing game** (Zhang et al., 2018). A localization metric from weakly-supervised grounding: take the single highest-response location in a saliency map and ask whether it falls inside the ground-truth region. Unlike Dice it is threshold-free and carries no geometric penalty against small targets, so it separates "did the model point at the lesion" from "did it draw the right boundary." Its chance level still scales with target size, so we always report lift over a per-bin chance baseline rather than the raw hit rate.

**Wilcoxon signed-rank test** (Wilcoxon, 1945). A non-parametric paired test. Because each patient is scored under both conditions, comparisons are paired; and because per-patient Dice is bounded, skewed, and frequently near zero for small lesions, a paired *t*-test's normality assumption is not safe. Wilcoxon ranks the absolute paired differences and tests whether positive and negative ranks are balanced.

**Benjamini-Hochberg FDR correction** (Benjamini & Hochberg, 1995). With 171 accumulated tests at α=0.05, roughly nine "significant" results would be expected from noise alone. BH sorts the $m$ p-values ascending and rejects the largest $i$ for which $p_{(i)} \le \frac{i}{m}\alpha$, controlling the expected *proportion* of false discoveries among rejections. It is less conservative than Bonferroni, which is appropriate here because these tests are positively correlated (they share a baseline arm).

**Matched-pairs rank-biserial correlation and bootstrap intervals.** Effect size for Wilcoxon, ranging from −1 to +1. Reported alongside every p-value because with n=20–32 per bin a result can be significant yet negligible; Section 7 flags such cases explicitly. Confidence intervals on mean paired differences use a percentile bootstrap (10,000 resamples, seeded) rather than a normal-theory interval, for the same distributional reason Wilcoxon is used.

### 5.2 Code components

All code was written for this project; see [`src/README.md`](src/README.md) for a file-by-file listing with line counts. The major components:

| Component | Files | Role |
|---|---|---|
| **Data pipeline** | `preprocess.py`, `text_encoder.py` | NIfTI loading, per-modality z-scoring inside the brain mask, resampling to 128³, true native-resolution lesion volumes for size binning; PubMedBERT embedding of all text variants. |
| **Datasets** | `dataset.py`, `dataset_rq2.py`, `dataset_rq4.py`, `dataset_pprime.py` | Patch samplers for each experimental condition (region-labeled, size-conditioned, scale-matched) plus the full-volume segmentation loader for P′. `region_mask()` here is the single definition of ET/TC/WT used everywhere. |
| **Model** | `model.py` | `TextVolumeAligner`: MONAI 3D ResNet-10 volume encoder plus a linear text projection into a shared 256-d L2-normalized space. |
| **Localization** | `localize.py` | `sliding_window_heatmap()`: sweeps the encoder across the volume, accumulating per-voxel cosine similarity to a query. Supports querying at a different physical window size than the model was trained at, which is what makes the whole RQ3/RQ3b/RQ3c/RQ12 window sweep possible without retraining. |
| **Training** | `train_baseline.py`, `train_rq2/4/5/6/7.py`, `train_pprime_supervised.py` | One script per experimental arm, each taking `--seed` to control the train/val split for cross-seed replication. |
| **Evaluation** | `evaluate_rq1.py` and 10 siblings | `evaluate_rq1.py` defines `otsu_threshold()`, `dice_iou()` and `size_bin()`, which every other evaluation — including the supervised P′ — imports, so all arms are scored by literally the same code. |
| **Diagnostics** | `test_rq4_shortcut_hypothesis.py`, `test_rq6_hub_bias.py`, `compute_chance_baseline.py`, `sanity_check_localize.py` | Noise probes, chance-level control, and the pre-flight check that the heatmap scores higher inside the true region than outside. |
| **Analysis** | `analyze_full_family.py`, `analyze_seed_replication.py`, `analyze_rq7_multiseed.py`, and 5 others | Recompute every statistic directly from the saved per-patient CSVs, so no number in this report is transcribed by hand. |
| **Figures** | 9 `make_figure*.py` scripts | Every figure regenerates from the result CSVs. |

Two recurring design decisions are worth naming. First, **evaluation scripts import their metric definitions rather than redefining them**, which is what makes cross-experiment comparison meaningful. Second, **several scripts carry a built-in correctness gate**: `evaluate_rq8_compositionality.py` asserts its "original" condition reproduces RQ1's CSV exactly, and `evaluate_grounding_sweep.py` at window 32 must reproduce RQ11's — both caught real bugs during development.

## 6. Core Result: Size-Stratified Localization Failure

**Note on multiple comparisons.** Sections 6 and 7 together report many paired significance tests (171 in total by the end of Section 7: region × size-bin × comparison, across 8 research questions). At uncorrected α=0.05, that volume of testing would be expected to produce a small number of spurious "significant" results by chance alone. We therefore apply Benjamini-Hochberg FDR correction across the full accumulated family of tests and report both the raw p-value and BH-adjusted q-value wherever a specific claim rests on statistical significance; any result that is significant raw but does not survive correction is explicitly flagged as such rather than presented as a finding.

### 6.1 P′: validating the pipeline against a previously-studied problem

Every result in this report is produced by one pipeline — our preprocessing, our split, our `dice_iou()`, our size terciles. If any of those carried a bug, "small lesions localize badly" would be unfalsifiable from the inside: we would have no way to distinguish a real grounding failure from, say, a misaligned mask or a broken metric. Two checks address this, an internal one and an external one.

**Internal: does the alignment learn anything?** Full run, 296 train / 73 val patients, 30 epochs: validation accuracy on the 4-way ET/TC/WT/NONE classification rose from 0.51 to a peak of 0.671 (epoch 26), finishing at 0.626, against a chance level of 0.25. The pipeline learns a genuine, non-trivial text-volume alignment signal.

**External (P′): does the same machinery reproduce published results on a problem someone else has already solved?** That problem is supervised BraTS tumor segmentation, scored publicly by the MICCAI challenge since 2012. We trained a plain 3D U-Net (Ronneberger et al., 2015) reusing, unchanged, the same preprocessed volumes, the same seed-0 split, the same `region_mask()` definitions and the same `dice_iou()` — changing only the model and the supervision signal.

| Region | P′ Dice (ours) | Published BraTS2020 range | In range? |
|---|---|---|---|
| ET | 0.758 | 0.70 – 0.80 | yes |
| TC | 0.812 | 0.80 – 0.87 | yes |
| WT | 0.851 | 0.86 – 0.89 | 0.009 below |

Two of three regions land inside the published range and the third is 0.009 short of it, which is what a single small U-Net on 128³ resampled volumes should look like against leaderboard entries that use native resolution, ensembling and test-time augmentation. **The shared machinery is therefore sound, and the low absolute Dice in Sections 6.2–7 is a property of text-conditioned localization rather than a defect in the plumbing underneath it.**

**P′ also answers a question no text-conditioned arm could.** Because it is scored by the identical size terciles, its large/small ratio is directly comparable to RQ1's — which separates "text-conditioned grounding fails on small lesions" from "small lesions are simply hard for everything on this data."

| Region | P′ supervised L/S | RQ1 text-conditioned L/S (Otsu) | RQ1 (oracle threshold) |
|---|---|---|---|
| ET | **1.3×** | 15.0× | 14.4× |
| TC | **1.3×** | 9.2× | 5.6× |
| WT | **1.2×** | 4.9× | 2.4× |

A fully supervised model on this exact data barely degrades at all: 0.636 → 0.858 Dice across ET's size range, against the text-conditioned model's 0.010 → 0.149. **Small lesions are not intrinsically unlearnable here, and Dice's geometric size penalty is not large enough to explain the collapse — a supervised model absorbs both and still scores 0.64 on the smallest enhancing tumors.** This is the strongest evidence in the report that the failure is specific to text-conditioned localization, and it rules out the most obvious deflationary reading of the entire project.

![P′ validation. Left to right: enhancing tumor, tumor core, whole tumor. Solid bars are the supervised U-Net, faded bars the text-conditioned baseline, both on the same held-out patients under the same Dice implementation. The supervised model's large/small ratio is 1.2–1.3× against the text-conditioned 4.9–15.0×.](figures/fig_pprime_size.png){width=98%}

### 6.2 RQ1: Size-stratified localization

| Region | Small Dice | Medium Dice | Large Dice | Large/Small ratio |
|---|---|---|---|---|
| ET | 0.010 ± 0.009 (n=21) | 0.037 ± 0.015 (n=23) | 0.149 ± 0.078 (n=23) | 14.9× |
| TC | 0.019 ± 0.012 (n=27) | 0.059 ± 0.022 (n=23) | 0.176 ± 0.062 (n=23) | 9.3× |
| WT | 0.057 ± 0.026 (n=32) | 0.137 ± 0.036 (n=21) | 0.281 ± 0.049 (n=20) | 4.9× |

The degradation is monotonic and severe across **all three independently-defined tumor subregions**, holding on a held-out validation set never seen during training. This directly and quantitatively confirms the small-lesion grounding failure documented qualitatively in recent 3D medical VLM literature (Section 3). Absolute Dice is modest throughout (a lightweight ResNet-10 with an unsupervised Otsu threshold, not a competitive segmentation model) — the finding is about the *relative* size-dependent collapse. Section 6.1 bounds how much of that modesty is the task setup rather than the pipeline: the same data, split and metric under dense supervision reach 0.76–0.85 Dice.

\newpage

![Dice vs. true lesion volume, model (blue) vs. a chance/random-heatmap control (gray), one panel per subregion. Both climb with volume, but the model sits well above chance throughout.](figures/fig1_dice_vs_volume.png){width=95%}

**Statistical validation — is this just a Dice artifact?** Dice is known to penalize small structures more harshly than large ones for *any* predictor, purely as a geometric property of the metric (a single misclassified voxel costs a tiny lesion far more of its Dice score than it costs a large one). To check whether the RQ1 collapse is a real model failure rather than this artifact, we computed a **chance baseline**: pure random-noise heatmaps run through the identical Otsu-threshold-and-Dice pipeline, on the same 73 validation patients.

The chance baseline collapses with size too — as expected, since this reflects the metric, not the model. Spearman correlation between Dice and log-volume is extremely strong for *both* the model (ρ=0.97–0.98) and chance (ρ=0.999–1.000, i.e. almost perfectly deterministic), confirming Dice's inherent size-dependence. The question that actually isolates the model's behavior is the **lift over chance** (model Dice ÷ chance Dice) at each size bin:

| Region | Small lift | Medium lift | Large lift |
|---|---|---|---|
| ET | 10.8× | 10.7× | 13.3× |
| TC | 8.4× | 8.5× | 9.4× |
| WT | 6.6× | 6.7× | 7.9× |

This is the more careful finding: the model's *relative* advantage over chance is roughly **constant** across size bins (within each region, small/medium/large lift are all within ~25% of each other) — the model isn't disproportionately losing signal on small lesions relative to a null baseline. What collapses is **absolute** localization quality: even with a consistent ~7-13× advantage over random guessing, small-lesion Dice (0.010–0.057) is nowhere near clinically usable, while large-lesion Dice (0.149–0.281), though still modest, is at least in a range serious methods report. So the honest claim is not "the model is uniquely broken on small lesions relative to its own baseline" — it's "even a consistent relative advantage over chance is not enough to produce usable absolute localization when the target is small," which is arguably the more clinically relevant framing anyway: a radiologist doesn't care whether a bad prediction is bad in an absolute or relative sense.

\newpage

![Qualitative example: T1ce slice with the model's predicted heatmap (hot colormap) and the true ET boundary (cyan). Left: a large lesion, where the heatmap's peak response aligns closely with the true tumor. Right: a small lesion, where the true tumor is a tiny dot completely swallowed by one oversized block of "high response" — a direct visual illustration of the fixed-patch-resolution problem discussed below.](figures/fig3_example_overlays.png){width=95%}

The qualitative example above makes the mechanism visible: the visible blockiness in both heatmaps is the sliding-window's fixed 32³ receptive field. For the large lesion it happens to roughly match the tumor's scale, so the peak response block and the true boundary line up reasonably well. For the small lesion, the entire true region fits inside a small fraction of a single response block — the window simply cannot resolve anything finer than its own size, regardless of how correct or incorrect its content judgment is.

**Robustness check: does this hold on a different split?** Every result up to this point uses one fixed 80/20 split (seed 0). To check this isn't an artifact of that particular split, we retrained and re-evaluated the entire alignment baseline from scratch on two additional independent random splits (seeds 1 and 2, same hyperparameters, same protocol).

| Region | Bin | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|---|
| ET | small | 0.0100 | 0.0083 | 0.0079 | 0.0087 ± 0.0009 |
| ET | medium | 0.0374 | 0.0375 | 0.0285 | 0.0345 ± 0.0042 |
| ET | large | 0.1488 | 0.1223 | 0.0981 | 0.1231 ± 0.0207 |
| TC | small | 0.0192 | 0.0220 | 0.0175 | 0.0196 ± 0.0019 |
| TC | medium | 0.0590 | 0.0738 | 0.0601 | 0.0643 ± 0.0068 |
| TC | large | 0.1756 | 0.2254 | 0.1477 | 0.1829 ± 0.0321 |
| WT | small | 0.0568 | 0.0445 | 0.0474 | 0.0496 ± 0.0053 |
| WT | medium | 0.1367 | 0.1220 | 0.1333 | 0.1307 ± 0.0063 |
| WT | large | 0.2809 | 0.2467 | 0.2576 | 0.2617 ± 0.0142 |

**The monotonic small < medium < large pattern holds exactly in all 3 seeds, for all 3 regions** (9 of 9 replications). Absolute Dice values shift somewhat between splits (expected, given different held-out patients and a modest ~73-patient validation set each time), but the core relationship this project is built around is not an artifact of one particular train/val split.

### 6.3 RQ11: How much of the collapse is grounding failure, and how much is the threshold?

Sections 6.1-6.2 measure localization through a two-stage pipeline: a continuous similarity heatmap, then an unsupervised Otsu threshold converting it to a binary mask. Every Dice number reported so far is a property of *both* stages. The chance-level control in Section 6.2 does not separate them — a random heatmap is binarized by the same Otsu step, so a thresholding pathology would appear in the model and the control alike and cancel out of the lift ratio.

This matters because Otsu picks its cut point from each volume's own intensity histogram, with no reference to the query or to plausible lesion size. If it emits a roughly constant-size blob regardless of the target, small lesions would score badly for a reason unrelated to text-conditioned grounding. Our result CSVs could not answer this: they never recorded how large the predicted mask actually was.

We therefore recomputed the heatmap once per (patient, region) with the identical frozen baseline and 32³/stride-16 protocol, then binarized it five ways: Otsu (the published protocol), fixed top-10%/5%/1% of voxels, and an **oracle-volume** threshold taking exactly as many voxels as the ground-truth mask contains. The last removes threshold calibration from the problem entirely and asks only how well the heatmap *ranks* voxels; it uses ground truth and is therefore a diagnostic upper bound, never a deployable method.

**Otsu is badly miscalibrated, and miscalibrated in the wrong direction.**

| Region | Bin | GT volume | Otsu predicted | Over-prediction | % of imaged volume |
|---|---|---|---|---|---|
| ET | small | 4,124 mm³ | 920,036 mm³ | 223.1× | 10.3% |
| ET | large | 50,716 mm³ | 718,728 mm³ | 14.2× | 8.1% |
| TC | small | 10,213 mm³ | 1,105,667 mm³ | 108.3× | 12.4% |
| TC | large | 84,708 mm³ | 937,076 mm³ | 11.1× | 10.5% |
| WT | small | 38,995 mm³ | 1,371,023 mm³ | 35.2× | 15.4% |
| WT | large | 164,364 mm³ | 1,022,709 mm³ | 6.2× | 11.5% |

Otsu returns a near-constant 8-15% of the imaged volume whatever the target's size. Worse, the correlation between true and predicted volume is **negative in all three regions** (ET ρ=−0.374, p=0.0018; TC ρ=−0.258, p=0.028; WT ρ=−0.475, p=2.2×10⁻⁵), where a calibrated predictor would approach +1. The thresholding step assigns *larger* masks to *smaller* lesions, mechanically manufacturing part of the size effect Section 6.2 attributes to grounding.

**The cost of that step is large and paired-significant in 8 of 9 bins.**

| Region | Bin | Otsu Dice | Oracle-volume Dice | Gain | p (Wilcoxon) |
|---|---|---|---|---|---|
| ET | small | 0.0100 | 0.0254 | 2.6× | 0.43 (n.s.) |
| ET | medium | 0.0374 | 0.1495 | 4.0× | 0.038 |
| ET | large | 0.1488 | 0.3652 | 2.5× | 2.4×10⁻⁷ |
| TC | small | 0.0192 | 0.0922 | 4.8× | 0.0089 |
| TC | large | 0.1756 | 0.5175 | 2.9× | 4.8×10⁻⁷ |
| WT | small | 0.0568 | 0.2600 | 4.6× | 2.0×10⁻⁶ |
| WT | medium | 0.1367 | 0.5162 | 3.8× | 9.5×10⁻⁷ |
| WT | large | 0.2809 | 0.6218 | 2.2× | 1.9×10⁻⁶ |

This revises a claim made in Section 6.2. We attributed modest absolute Dice to "a lightweight ResNet-10 baseline with an unsupervised Otsu threshold" — the measurement now shows the threshold, not the backbone, carried most of that cost. Whole-tumor large-lesion Dice is 0.622 under proper binarization, not 0.281. The single exception is ET-small, the only bin where better thresholding does *not* help significantly (p=0.43): there the heatmap's ranking is itself poor, so no cut point recovers it.

**The size collapse survives, at reduced magnitude.**

| Region | L/S ratio (Otsu, as reported in 6.2) | L/S ratio (oracle-volume) | Spearman(Dice, volume) Otsu → oracle |
|---|---|---|---|
| ET | 15.0× | **14.4×** | 0.970 → 0.835 |
| TC | 9.2× | **5.6×** | 0.978 → 0.770 |
| WT | 4.9× | **2.4×** | 0.968 → 0.781 |

Enhancing tumor's collapse is essentially entirely genuine; tumor core's and whole tumor's are roughly half thresholding artifact. The honest headline is therefore **2.4-14.4× degradation after controlling for binarization**, not 5-15×. The relationship remains monotonic, strongly positive, and present in every region under every one of the five binarization rules tested — it is the magnitude, not the existence, that was overstated.

**A threshold-independent metric isolates where the real failure lives.** Dice conflates "did the model point at the lesion" with "did it draw the right boundary." We therefore also report the **pointing game** (Zhang et al., 2018): is the highest-response location inside the true mask? It is free of Dice's geometric size penalty. Its chance level still scales with target size, so we compare against a per-bin chance baseline (mean GT voxels ÷ total voxels) with an exact binomial test.

**First, a correction to how that peak is located.** A sliding window of stride $s$ gives every voxel the mean of the windows covering it, and all voxels inside the same $s^3$ block are covered by an identical window set — so the heatmap is *piecewise-constant over $s^3$ blocks*, not smooth. At our published 32³/stride-16 protocol that block is 16³ = 4096 voxels, roughly 30 mm across, and `argmax` silently returns the block's **corner** in array order rather than anything peak-like. We verified this directly: the median count of voxels tied at the maximum is exactly 4096 at stride 16 and exactly 512 at stride 8, matching $s^3$ in both cases. We therefore locate the peak at the **centroid of the tied-maximum plateau** and report both rules below, since the naive rule biases hit rates down and distances up.

| Region | Bin | n | Hits (corrected) | Hit rate | Chance | Lift | p | Median distance |
|---|---|---|---|---|---|---|---|---|
| ET | small | 21 | **0** | 0.000 | 0.0005 | **0.0×** | 1.00 | 23.7 mm |
| ET | medium | 23 | 3 | 0.130 | 0.0018 | 74.5× | 9.3×10⁻⁶ | 9.4 mm |
| ET | large | 23 | 8 | 0.348 | 0.0057 | 61.2× | 4.9×10⁻¹³ | 2.9 mm |
| TC | small | 27 | 3 | 0.111 | 0.0011 | 97.1× | 4.3×10⁻⁶ | 12.0 mm |
| TC | medium | 23 | 8 | 0.348 | 0.0035 | 99.0× | 1.1×10⁻¹⁴ | 1.9 mm |
| TC | large | 23 | 16 | 0.696 | 0.0095 | 73.3× | 9.9×10⁻²⁸ | 0.0 mm |
| WT | small | 32 | 17 | 0.531 | 0.0044 | 121.6× | 4.1×10⁻³² | 0.0 mm |
| WT | medium | 21 | 18 | 0.857 | 0.0105 | 81.9× | 2.9×10⁻³³ | 0.0 mm |
| WT | large | 20 | 17 | 0.850 | 0.0184 | 46.2× | 3.5×10⁻²⁷ | 0.0 mm |

The correction matters most where the model was already doing well: whole-tumor small goes from 6/32 hits to 17/32, and tumor-core large from 13/23 to 16/23. **The headline claim is unaffected: enhancing-tumor small remains 0 hits in 21 patients under both rules.** The old first-argmax numbers are retained in the analysis log for comparison.

<!-- superseded rows, kept for reference against the first-argmax rule:
| ET | small | 21 | 0 | 0.000 | 0.0005 | 0.0× | 1.00 | 27.5 mm |
| ET | large | 23 | 6 | 0.261 | 0.0057 | 45.9× | 3.1×10⁻⁹ | 2.4 mm |
| TC | large | 23 | 13 | 0.565 | 0.0095 | 59.6× | 5.3×10⁻²¹ | 0.0 mm |
| WT | small | 32 | 6 | 0.188 | 0.0044 | 42.9× | 5.7×10⁻⁹ | 8.7 mm |
| WT | large | 20 | 16 | 0.800 | 0.0184 | 43.5× | 7.9×10⁻²⁵ | 0.0 mm |
-->

Chance-corrected lift is roughly **uniform at 46-122× across every region and size bin** — echoing Section 6.2's finding that lift over chance is approximately constant, but now with a metric that has no built-in size bias. The exception is stark and specific: **for small enhancing tumor the model is at exact chance, 0 hits in 21 patients, with its peak response a median 23.7 mm from the nearest lesion voxel.** That is not a boundary-delineation failure; the model is not pointing at the lesion at all. This is the sharpest statement of the failure this project can make, and it is confined to one region×bin rather than being the smooth monotonic collapse the Dice framing suggests. Section 7.12 returns to this bin and shows it is the one place the smaller-window intervention genuinely helps.

**One deployable consequence.** The fixed top-1% threshold (`pct99`) needs no ground truth and beats Otsu everywhere, by 3-4× for ET (0.0436/0.1774/0.3672 vs 0.0100/0.0374/0.1488). For ET it even beats the oracle-volume threshold in all three bins — when a heatmap's ranking is unreliable, a mask larger than the true lesion is Dice-optimal, because concentrating into exactly the right *number* of voxels is only rewarded if they are the right *ones*. Replacing Otsu with a fixed percentile is a free protocol improvement, and unlike the oracle it is usable at inference time.

## 7. Ablation Studies

RQ1 established that the failure is real. This section systematically ablates candidate explanations and fixes — text-side conditioning, inference-side window size, and training-side retraining — to find out whether the failure is linguistic, architectural, or trainable away. Figure below previews the full decision tree and how each ablation's outcome motivated the next.

![Roadmap of every experiment in the project, from the P′ validation and RQ1's core finding through the three branches — text-side, inference-side, training-side — with each box's one-line outcome. Green = confirmed or helped, red = hurt or ruled out, amber = partial or superseded, blue = a neutral finding. The centre column is the one to read carefully: RQ3b/RQ3c is amber not because it failed but because its evaluation confounded window size with tiling convention, and RQ12 below it is the control that separated the two and confirmed the effect.](figures/fig_roadmap.png){width=98%}

### 7.1 RQ2: Size-conditioned prompting mitigation

Mechanism: at training time, each ET/TC/WT patch is labeled with its lesion's *true* size bin (known from ground truth) and trained against a size-specific text description (10-way classification: 3 regions × 3 sizes + NONE). At evaluation time, since true size is unknown a priori, we query with all three size-phrasings per region and take the voxel-wise maximum response (a deployable ensemble, not test-time label leakage).

Training: validation accuracy on the finer-grained 10-way task reached 0.552 (chance = 0.10) after 30 epochs.

> **The positive half of this section is withdrawn in Section 7.12.** The improvements below are Otsu-scored; under every calibrated binarizer they vanish (+0.006 at top-1%, p=0.11). What survives is the negative half — that size-conditioned prompting worsens small enhancing tumor and widens the ET size gap — which also replicates across three seeds (Section 7.10).

| Region | Bin | RQ1 (baseline) | RQ2 (size-conditioned) | Δ |
|---|---|---|---|---|
| ET | small | 0.010 | 0.008 | −0.002 |
| ET | medium | 0.037 | 0.038 | +0.001 |
| ET | large | 0.149 | 0.191 | **+0.042** |
| TC | small | 0.019 | 0.027 | +0.008 |
| TC | medium | 0.059 | 0.095 | **+0.036** |
| TC | large | 0.176 | 0.252 | **+0.076** |
| WT | small | 0.057 | 0.059 | +0.002 |
| WT | medium | 0.137 | 0.147 | +0.010 |
| WT | large | 0.281 | 0.321 | +0.040 |

\newpage

![RQ1 (baseline) vs RQ2 (size-conditioned) Dice by region and size bin, error bars = 1 std.](figures/fig2_rq1_vs_rq2.png){width=95%}

**Statistical validation.** A paired Wilcoxon signed-rank test (matched by patient, RQ1 vs RQ2 Dice) at each region × size bin gives a precise picture, and it's more nuanced than the raw means alone suggest:

| Region | Small bin | Medium bin | Large bin |
|---|---|---|---|
| ET | **significantly worse** (p=0.010) | no significant change (p=0.23) | significantly better (p<0.0001) |
| TC | significantly better, tiny effect (p<0.0001, +0.008) | significantly better (p<0.0001, +0.036) | significantly better (p<0.0001, +0.076) |
| WT | significantly better, tiny effect (p=0.017, +0.002) | not significant (p=0.050, borderline) | significantly better (p=0.004) |

**Finding: the mitigation did not work for its intended purpose, and the effect is not uniform across regions.** For ET — the smallest and hardest region, and arguably the one that matters most clinically — size-conditioned prompting made small-lesion localization *significantly worse* (p=0.010), the opposite of its goal. For TC and WT, small-bin Dice did improve with statistical significance, but the effect sizes are clinically negligible (+0.008 and +0.002 respectively) — real, but not meaningful. In relative terms the large/small gap for ET actually *widened* (14.9× → 23.9×). Meanwhile every region shows a real, often large, improvement at the medium/large end (TC medium +61% relative, TC large +43% relative, both p<0.0001). So the mitigation is doing something — it is reliably better at medium/large lesions — but that something is not solving, and for the hardest region actively worsens, the specific small-lesion problem it was designed to fix.

**Why, most likely:** the sliding-window patch is a fixed 32³ voxels, which at this resampled resolution (128³ from an original ~240×240×155 grid) spans roughly 60×60×39mm physically. A "small" ET lesion can be as little as 32mm³ — a handful of voxels — meaning even a perfectly-worded "small lesion" text query is matched against a window enormously larger than the lesion itself. The lesion's signal is diluted by surrounding normal tissue inside the patch regardless of what text conditions the query. This points to a **resolution/architecture bottleneck, not a language bottleneck** — you cannot prompt your way out of a fixed receptive field that's larger than the target.

### 7.2 RQ3: Does the receptive field actually explain it? (naive multi-scale windowing)

RQ2's explanation is a claim, not yet a test. If a fixed 32³ receptive field really is the bottleneck, then evaluating the *same, frozen* RQ1 model at a smaller physical window (resized to the model's 32³ input before encoding) should help, with no retraining at all. We swept three window sizes (16³, 32³, 64³, each with stride = window/2) and combined them via voxel-wise maximum.

| Region | Small bin | Medium bin | Large bin |
|---|---|---|---|
| ET | not significant (p=0.43) | significantly worse (p=0.0001) | significantly worse (p=0.016) |
| TC | not significant (p=0.40) | not significant (p=0.12) | significantly worse (p=0.014) |
| WT | **significantly better** (p=0.033) | significantly worse (p=0.001) | significantly worse (p=0.006) |

**Naive multi-scale ensembling mostly hurts.** One of nine bins improved; five got significantly worse; three showed no significant change. (All of these hold after BH-FDR correction across the full test family — q-values track the raw p-values closely here.) This seems to contradict RQ2's explanation — but a quick smoke test on a single patient hinted at why it might not: the 16³ window alone produced a much higher peak similarity score (0.259) than the 32³ baseline (0.061), while 64³ produced uniformly *negative* scores everywhere (the model was never trained to interpret a resized-down 64³ crop, so it just never fires). Voxel-wise max across scales means that wherever *any* scale spuriously spikes, that noise wins — and a scale the model can't interpret meaningfully (64³) is exactly the kind of scale that produces uncalibrated, noisy responses.

### 7.3 RQ3b: Isolating the effect — one smaller window, no ensembling

To separate "does a smaller receptive field help" from "does naively combining multiple scales help," we re-ran evaluation using *only* the 16³ window (no ensembling with 32³ or 64³), still on the same frozen RQ1 model, still no retraining.

| Region | Small Dice | Medium Dice | Large Dice |
|---|---|---|---|
| ET | 0.011 (+11%) | 0.039 (+6%) | 0.160 (+8%) |
| TC | 0.023 (+21%) | 0.069 (+17%) | 0.198 (+13%) |
| WT | 0.072 (+26%) | 0.156 (+13%) | 0.343 (+22%) |

*(percentages are relative improvement over the RQ1 32³ baseline)*

**All 9 region×bin combinations improved at raw p<0.05 — but this needs the multiple-comparisons correction applied honestly.** After BH-FDR correction across the full accumulated test family, **TC's and WT's improvements survive (6/6, q<0.05 throughout)**, but **ET's three bins do not** (q=0.052, 0.052, 0.064 — just above the corrected threshold). ET is also the region the report has repeatedly flagged as the smallest, hardest, and clinically most important — so the one region where this fix matters most is exactly the one where the statistical evidence is weakest. The honest claim is: a smaller window robustly helps TC and WT localization at every size, and probably helps ET too, but that specific claim doesn't clear a properly corrected significance bar with n=21-23 patients per bin.

Where the win is real (TC, WT), it also isn't free computationally. The 16³/stride-8 sweep does 3,375 forward passes per volume versus the 32³/stride-16 baseline's 343 — a 9.8× increase — and matches in wall-clock: the RQ1 evaluation job ran in 2m47s, the RQ3b job in 19m17s, roughly 7×. "No retraining needed" is accurate; "free" is not, and we shouldn't have said it that way.

Even with those caveats, the large/small *ratio* barely moves either way (ET 14.9×→14.4×, TC 9.3×→8.6×, WT 4.9×→4.8×) — this is a real lift to absolute quality for two of three regions, not a fix for the underlying relative size gap.

> **Read this section together with Section 7.11.** Every number above is scored under Otsu, and RQ12 later shows the smaller-window advantage is specific to that binarizer and reverses under better-calibrated ones. The statistics here are correct as computed; what they measure turned out not to be what we thought.

### 7.4 RQ3c: How far does "smaller is better" go?

RQ3b only compared one alternative (16³) against the 32³ baseline. To see whether the trend continues, reverses, or plateaus, we extended the same isolated-window evaluation (no ensembling, no retraining, same frozen RQ1 model) to three smaller sizes: 12³ (stride 6), 8³ (stride 8), and 6³ (stride 6; the latter two use non-overlapping tiling rather than the 50%-overlap convention used elsewhere, for compute reasons — flagged here rather than left implicit).

![Mean Dice (pooled across size bins) vs. sliding-window size, one line per region. The trend from RQ3b continues through 12³/8³, then plateaus for ET/TC at 6³ while WT keeps climbing.](figures/fig5_window_curve.png){width=80%}

The trend **continues in the same direction through 8³, with no plateau or reversal**:

| Region | 32³ (baseline) | 16³ | 12³ | 8³ | 6³ |
|---|---|---|---|---|---|
| ET large | 0.149 | 0.160 | 0.161 | 0.177 | 0.177 |
| TC large | 0.176 | 0.198 | 0.208 | 0.242 | 0.245 |
| WT large | 0.281 | 0.343 | 0.360 | 0.426 | 0.437 |

More importantly, re-running the full BH-FDR correction across all tests accumulated up to this point, **ET's statistical picture actually improves as the window shrinks further**: at 12³, 2 of ET's 3 bins now survive correction (only ET-large is borderline, q=0.052); at 8³, **all 3 of ET's bins survive**, along with every other region and bin (18/18 at this window size and the one below it). The region that couldn't clear a corrected significance bar at 16³ becomes the region with the cleanest signal at 8³.

**Pushing one step further to 6³, the trend plateaus for ET and TC specifically.** A paired Wilcoxon test between 8³ and 6³ shows no significant difference for any ET or TC bin (all 6 bins n.s., p=0.13–0.95), but WT continues to improve significantly in all 3 bins (p<0.05 throughout, including p=0.0006 for WT small). A plausible reason: WT lesions are the largest of the three regions by a wide margin (median ~90,740mm³ vs. ET's ~16,971mm³ and TC's ~33,808mm³), so even an 8³ or 6³ window is still comparatively small relative to WT's typical physical extent, leaving more room to benefit from further shrinking, while ET and TC — already much smaller lesions — have apparently reached the point where the window is no longer the binding constraint. The "smaller is better" trend does have a floor, and it isn't the same floor for every region.

> **Read this section together with Section 7.11.** Two problems with the curve above are resolved there. First, it is Otsu-scored throughout, and Otsu turns out to *understate* the smaller-window benefit by 2-3× relative to calibrated rules. Second — and more seriously — the 8³ and 6³ points also changed the stride convention, so they vary two things at once and are not comparable to the three points before them. Section 7.11 adds the missing overlap-matched control; the direction of this curve survives, but the plateau claim below does not.

### 7.5 RQ4: Training with scale-matched patches

RQ3b changes only *evaluation*. The more ambitious version of the same idea is to also *train* with patches whose physical crop size matches the declared size bin (small→16³, medium→32³, large→64³, each resized to the canonical 32³ input), combined with RQ2's size-conditioned text (10-way classification), then evaluate with each size-phrasing queried at its matched scale and combined via max.

Training reached a *better* classification accuracy than RQ2 (0.668 vs. 0.552 best val accuracy on the analogous 10-way task) — the model clearly learned to use scale as a signal. But localization got **significantly worse than the RQ1 baseline in all 9 bins (p<0.0001 throughout, and all 9 survive BH-FDR correction)**. We also directly verified the comparison to RQ2 rather than asserting it: a paired test confirms RQ4 is **significantly worse than RQ2 in all 9 bins as well (p<0.0001 throughout, all surviving correction)**.

| Region | Small Dice | Medium Dice | Large Dice |
|---|---|---|---|
| ET | 0.005 | 0.018 | 0.064 |
| TC | 0.012 | 0.036 | 0.098 |
| WT | 0.048 | 0.106 | 0.178 |

**Better classification accuracy did not transfer to better localization — and we tested why, rather than just speculating.** The hypothesis: since each size-conditioned class was trained on patches resized in a systematically different direction (16³ crops upsampled/blurred, 64³ crops downsampled/detail-lost), the model might key off *resize-interpolation artifacts* correlated with the size label rather than genuine tumor content. We tested this directly: we fed **pure random Gaussian noise** — zero real anatomical content — through the identical three crop-then-resize pipelines used in training, and measured the trained RQ4 model's similarity between the resulting image embeddings and each size-conditioned text embedding.

| Noise pipeline (crop size) | Mean similarity to its own matching size-text |
|---|---|
| "small" (16³→32³) | −0.401 ± 0.010 |
| "medium" (32³ native) | +0.026 ± 0.010 |
| "large" (64³→32³) | +0.149 ± 0.008 |

A one-way ANOVA across the three noise groups gives F=59,672, p≈4×10⁻²⁵¹ — an essentially complete, non-overlapping separation, on pure noise with no tumor present. **This confirms the model is not relying on real content for a substantial part of its size signal.** The mechanism is more specific than we first guessed, though, and worth reporting precisely rather than rounding it off to the original hypothesis: it is not that each pipeline gets cleanly fingerprinted to its own label. Every noise pipeline — including "small" and "medium" — scores *highest* against the **"large"** text embedding (small-pipeline noise: −0.400 vs. "small" text, but +0.015 vs. "large" text; medium-pipeline noise: +0.023 vs. "medium" text, but +0.126 vs. "large" text). This looks like a generic bias toward the "large" class dominating almost regardless of input — an embedding-space hub or degenerate solution — rather than genuine per-scale artifact recognition. Either way, the conclusion is the same: RQ4's size classification is not reliably grounded in real tumor content, which is a sufficient and now directly-verified explanation for why its improved training-time accuracy failed to produce better localization. Combined with RQ3's finding that max-ensembling across scales amplifies whichever scale is noisiest, RQ4 appears to inherit both problems at once.

### 7.6 RQ6: Can the embedding hub actually be fixed?

RQ4 diagnosed a real problem (the "large"-class hub) rather than solving one. We attempted an actual fix: retrain the identical scale-matched setup with an added **uniformity regularizer** — a loss term that directly penalizes high pairwise cosine similarity among the projected text class embeddings, pushing them apart on the hypersphere so no single class can act as a generic attractor regardless of input content.

It worked, partially, and we verified each part of that claim rather than asserting it:

**The embeddings actually separated.** Mean pairwise cosine similarity among the text projections fell from ~0.97 (matching RQ4's near-total collapse) to **−0.10** within a few epochs of training, and stayed there. Re-running the exact noise probe from RQ4 confirms this is a real, not just numerical, change: **2 of 3 noise pipelines now correctly prefer their own matching label** (medium-pipeline noise scores highest on "medium" text, large-pipeline noise scores highest on "large" text), up from 0 of 3 in RQ4. Only the smallest-scale pipeline still shows residual bias toward "large."

**Localization genuinely improved over RQ4 — verified with paired tests, not eyeballed.** RQ6 beats RQ4 in **all 9 of 9 region×bin combinations**, every one significant (p<0.01, most p<0.001).

| Region | RQ4 small | RQ6 small | RQ4 medium | RQ6 medium | RQ4 large | RQ6 large |
|---|---|---|---|---|---|---|
| ET | 0.005 | 0.005 | 0.018 | 0.020 | 0.064 | 0.070 |
| TC | 0.012 | 0.016 | 0.036 | 0.046 | 0.098 | 0.120 |
| WT | 0.048 | 0.061 | 0.106 | 0.137 | 0.178 | 0.226 |

**But RQ6 still falls short of the plain RQ1 baseline in most bins.** Paired against RQ1, RQ6 is significantly *worse* in all of ET and TC (7 of 9 bins), and only statistically tied (not better) in WT small/medium, worse again in WT large. Fixing the embedding collapse was necessary but not sufficient.

**A follow-up isolates why, and the answer reverses our own expectation.** We hypothesized RQ4/RQ6's other diagnosed problem — RQ3's finding that max-ensembling across scales amplifies noise — was still holding RQ6 back, and that evaluating with *only* the correctly-matched single scale per patient (no ensembling) would do better. We tested this directly (using the true size bin as an oracle purely to isolate the mechanism, not as a deployable evaluation) and found the **opposite**: the single-scale version is significantly *worse* than RQ6's own 3-way ensemble in 8 of 9 bins, dramatically so for large lesions (roughly half the Dice: e.g. WT large 0.226 → 0.108).

This means ensembling across scales is not inherently harmful, as RQ3 seemed to show — it depends entirely on whether the underlying model was actually trained to produce calibrated responses at each scale. RQ1 was trained only at 32³ and evaluated out-of-distribution at 16³/64³ in RQ3, where ensembling amplified noise. RQ4/RQ6 were specifically trained across all three scales, and for them, ensembling captures genuine complementary information — most likely because a single window, even at the "correct" scale, cannot cover a large lesion's full spatial extent, while sweeping multiple scales and taking the max lets different parts of a large lesion register at whichever scale best captures them.

### 7.7 RQ5: Does the finding survive naturalistic language?

All of RQ1/RQ2's text was templated, textbook-style description. To rule out the entire size-collapse finding being some artifact of that specific phrasing, we rewrote the four base descriptions in naturalistic, radiology-report style (hedged, varied syntax — e.g. *"Post-contrast T1-weighted images demonstrate irregular nodular enhancement, favoring viable, high-grade tumor tissue"* rather than *"Enhancing tumor: region of active contrast enhancement..."*), retrained from scratch, and re-ran the identical RQ1 evaluation protocol.

Result on this single run: **essentially no difference.** 8 of 9 region×bin comparisons show no significant difference from the original templated-text RQ1 (paired Wilcoxon, all p>0.07), and Spearman correlation between Dice and log-volume is nearly identical (ρ=0.958 templated vs. 0.959 naturalistic). The one significant difference (TC large, p=0.0035) is a small improvement.

> **This conclusion is partly revised in Section 7.10.** Retrained under three seeds, naturalistic text is consistently *better* than templated (+0.040 pooled Dice, positive in all three runs), so "no difference" was a seed-0 artifact and this is a weaker robustness check than it appears here. What survives is the part that matters for the paper: the size collapse persists under naturalistic text in every seed, so the finding is not a templating artifact.

### 7.8 RQ7: Does the text encoder contribute anything at all?

Every ablation so far manipulated *what the text says*. RQ7 asks a blunter question: does it matter *what produces the embedding*? We retrained the identical architecture and protocol under four substituted text conditions, changing nothing else:

- **BERT-base** — general-domain rather than biomedical pretraining.
- **Random-init BERT (frozen)** — the PubMedBERT architecture with randomly initialized weights, never trained. Preserves BERT's embedding *geometry* but carries no learned semantics.
- **4 random orthonormal vectors** — pure class identifiers. No language, no geometry inherited from any encoder.
- **Random vectors at PubMedBERT's geometry** — random vectors resampled to match PubMedBERT's anisotropic covariance structure. Semantically empty, but geometrically identical to the real thing.

The last two are the decomposition that makes this informative: comparing PubMedBERT against *random vectors at its own geometry* isolates the contribution of **meaning** with geometry held fixed, while comparing orthonormal against anisotropic random vectors isolates the contribution of **geometry** with semantics held fixed at zero.

**The single-seed result looked dramatic and was largely wrong.** Pooled over all regions and bins, patient-paired tests reported random-init BERT *beating* PubMedBERT by +0.038 Dice at p≈6×10⁻³⁵, and BERT-base beating it by +0.015 at p≈4×10⁻¹⁵. Those p-values are not defensible. Each condition was trained exactly once, so the unit of randomization is the *training run*, not the patient; treating 213 patients from one run as 213 independent observations is pseudo-replication. The check that exposed it: retraining the *identical* PubMedBERT model under three seeds moves pooled Dice by 0.0086 on its own — comparable to several of the effects being claimed.

**Retrained under three seeds each, almost nothing survives.**

| Text condition | seed 0 | seed 1 | seed 2 | mean Δ | verdict |
|---|---|---|---|---|---|
| BERT-base (general domain) | +0.0151 | −0.0378 | +0.0078 | −0.0050 | sign flips → **noise** |
| Random-init BERT (frozen) | +0.0379 | +0.0283 | −0.0014 | +0.0216 | sign flips → **noise** |
| Random vectors, PubMedBERT geometry | −0.0072 | +0.0370 | +0.0085 | +0.0128 | sign flips → **noise** |
| **4 random orthonormal vectors** | −0.0140 | −0.0103 | −0.0020 | **−0.0087** | consistent, 2.0× noise floor |

Noise floor (baseline retrained, 3 seeds) = 0.0044 pooled Dice.

![RQ7 text-encoder ablations. Each dot is one independent training run; the vertical tick is the mean of the three. The grey band is the retraining noise floor, measured by retraining the *identical* PubMedBERT baseline under three seeds. Only the orthonormal-vector condition (orange) keeps its sign across all three runs; the other three change direction from run to run despite within-run p-values as small as 6×10⁻³⁵.](figures/fig_rq7_encoder.png){width=98%}

Three findings follow, and the first two are uncomfortable.

**Biomedical pretraining is worth nothing measurable here.** Swapping PubMedBERT for general-domain BERT-base produces an effect that changes sign across training runs. Whatever domain knowledge PubMedBERT encodes about enhancing tumor and peritumoral edema, this pipeline does not use it.

**Neither is language.** Replacing the encoder entirely with random vectors — no text, no tokenizer, no pretraining — is also indistinguishable from noise, provided those vectors carry PubMedBERT's anisotropic geometry. The one condition that *does* reliably underperform is the one that discards that geometry (orthonormal vectors, −0.0087, consistent across all three seeds). The decomposition is therefore stark: **the semantic content of the text contributes nothing detectable, while its embedding geometry contributes a small but reproducible amount.** Section 3 noted that our four class descriptions sit at ~0.99 pairwise cosine similarity; RQ7 shows the trainable projection head is not merely doing *most* of the discriminative work, it is doing essentially *all* of it, and it works about as well starting from arbitrary vectors as from meaning.

**The core finding is encoder-independent.** The large/small Dice ratio stays in the same range under every text condition and every seed (ET 10.6–18.2×, TC 7.3–17.6×, WT 4.6–5.6×). The small-lesion collapse is not a property of PubMedBERT; it survives replacing the language model with noise.

### 7.9 RQ8: Is the query functioning as language, or as a class identifier?

RQ7 shows the encoder can be replaced. RQ8 asks the complementary, more mechanistic question directly of the *trained* baseline: does it respond to linguistic structure at all? We built four manipulations of each region description and re-ran the frozen RQ1 model on all of them — evaluation only, no retraining:

- **negated** — the same clinical content under explicit negation ("*no* enhancing tumor is seen"). Preserves nearly every content word while inverting the meaning.
- **shuffled** — word order destroyed by a fixed-seed permutation. Holds the bag of words exactly constant, removes only syntax.
- **swapped** — the sentence frame kept, the region's head term replaced by a different region's.
- **generic** — contentless filler naming no anatomy or pathology at all ("a region of tissue"). The floor condition.

The "original" condition reproduces `rq1_localization_scores.csv` exactly, which is an automatic end-to-end correctness gate on the script.

**The model barely notices any of it.**

| Manipulation | Cosine to original, *after* the trained projection | Heatmap Spearman ρ vs. original |
|---|---|---|
| negated | 0.940 – 0.962 | 0.561 – 0.939 |
| shuffled | 0.962 – 0.974 | 0.753 – 0.996 |
| swapped | 0.954 – 0.981 | 0.793 – 0.965 |
| generic | 0.940 – 0.947 | 0.777 – 0.924 |

Every manipulation stays above 0.94 cosine to the original query *after* the trained text head — the component supposedly responsible for making these classes discriminable. The projection does not separate "enhancing tumor" from "no enhancing tumor" from "a region of tissue" in any strong sense.

![RQ8 compositionality probes. Spearman correlation between each manipulated query's heatmap and the true clinical query's, averaged over held-out patients. A model that composed meaning would drop sharply under negation and under word-order destruction; instead every bar sits between 0.56 and 1.00, and shuffling word order leaves whole-tumor behaviour essentially unchanged (ρ=0.996).](figures/fig_rq8_compositionality.png){width=98%}

**Contentless and wrong-referent text sometimes localizes better than the real description.** For whole tumor, the generic filler beats the true clinical description (+0.017 Dice) and the wrong-region term beats it by more (+0.033). Shuffling word order changes whole-tumor Dice by −0.0009 — not significant (q=0.09), i.e. **syntax is worth nothing for the largest region**. Negation does degrade performance everywhere, but that is the expected behaviour of a bag-of-words matcher encountering extra tokens, not evidence of understood polarity: if the model represented negation, the negated ET query should behave like the *background* class rather than like a slightly degraded ET query.

**Embedding distance does not predict behavioural change.** Across the 12 condition×region cells, the correlation between how far a manipulation moves the projected query and how much it changes localization is ρ=+0.41, p=0.19. If the embedding carried graded meaning, moving further should change behaviour more. It does not.

Taken with RQ7, this is the mechanistic version of the paper's title claim. The text query is not functioning as language; it is functioning as an **opaque class identifier that happens to be spelled in English**. That explains RQ5's null result (Section 7.7) far better than "the failure is robust to phrasing": swapping templated for naturalistic text changed little because *no* phrasing change matters much when the pathway is a lookup, not a parse.

### 7.10 Do the ablation conclusions replicate across training runs?

Section 6.2 replicated the *core* finding across three splits, but every ablation conclusion above rested on one training run each — exactly the trap RQ7 exposed. We retrained and re-evaluated RQ2, RQ4, RQ5 and RQ6 under seeds 0/1/2.

| Ablation | seed 0 | seed 1 | seed 2 | mean Δ | region×bin combinations replicating in all 3 |
|---|---|---|---|---|---|
| RQ2 (size-conditioned text) | +0.0227 | +0.0098 | +0.0258 | +0.0194 | 4 / 9 |
| RQ4 (scale-matched) | −0.0376 | −0.0466 | −0.0109 | −0.0317 | 6 / 9 |
| RQ5 (naturalistic text) | +0.0035 | +0.0528 | +0.0641 | +0.0401 | 6 / 9 |
| RQ6 (uniformity fix) | −0.0231 | −0.0378 | −0.0295 | −0.0301 | 7 / 9 |

All four hold their *direction* in all three runs, so the headline verdicts — RQ2 helps on average, RQ4 and RQ6 hurt — are not split artifacts. Three qualifications follow, and one is a correction.

**A correction to Section 7.7.** RQ5 was reported as showing "essentially no difference" from templated text. That was a seed-0 artifact. Across three runs naturalistic text is consistently *better* (+0.040 pooled, positive in all three seeds, 9× the noise floor), driven by TC and WT which replicate 3/3 in every size bin. The n=3 one-sample *t*-test is p=0.16 and therefore not significant on its own — with three runs it has almost no power — so the honest statement is: **consistent in direction and large relative to retraining noise, but not formally significant at this replication count.** What does *not* change is the conclusion that mattered: the size collapse persists under naturalistic text (L/S ratios 11.6–16.2× ET, 6.4–9.6× TC, 4.1–4.9× WT), so the finding is still not a templating artifact. RQ5 is a weaker robustness check than claimed and a stronger positive result than claimed.

**RQ2's per-bin story is the least stable.** Only 4 of 9 combinations replicate in all three seeds. Specifically, the headline claim that size-conditioned prompting *worsens* small enhancing tumor holds in 2 of 3 seeds (−0.0021, −0.0015, +0.0042), and the whole-tumor small bin flips outright (1/3). The robust part of RQ2 survives and is arguably sharper: size-conditioned prompting consistently *increases* the ET large/small ratio (24.3×, 23.0×, 16.3× versus the baseline's 15.0×, 14.7×, 12.4×) in every seed. It makes the size disparity worse even where it raises mean Dice.

**RQ4 and RQ6 are the most stable negative results in the project** (6/9 and 7/9), and both reduce the L/S ratio relative to baseline — they compress the size gap by degrading large-lesion performance, not by improving small-lesion performance.

### 7.11 RQ12: Is the smaller window's win real, and is Otsu measuring what we think?

Sections 7.3-7.4 crowned the isolated smaller window as this project's best intervention on the strength of Dice under Otsu. Two things about that verdict were never checked, and Section 10 flagged both as the most valuable outstanding follow-up. First, Dice conflates finding a lesion with outlining it, so a Dice win is consistent with the window merely tightening masks around lesions the model already found. Second, every Section 7 comparison shares the Otsu binarizer, which we argued made them internally fair — but Section 6.3 then showed Otsu is *anti*-correlated with lesion size, so it is not a neutral common factor at all.

We recomputed the heatmaps across the whole window sweep and scored each under all five binarization rules plus the corrected pointing game. Running the pipeline at 32³/stride-16 reproduces `rq11_threshold_confound_scores.csv` on 212 of 213 patients bit-for-bit (the one exception differs by 5×10⁻⁴ Dice, from a tie in Otsu's 256-bin histogram) — an end-to-end correctness gate on the generalization.

**A confound in the original sweep had to be removed first.** Section 7.4 noted in passing that 8³ and 6³ used non-overlapping tiling rather than the 50%-overlap convention used at 32³/16³/12³, "for compute reasons." That parenthetical turns out to be load-bearing: those two points change *stride as well as window size*, so they were never comparable to the rest of the curve. We therefore added an 8³/stride-4 run that holds the overlap convention fixed.

**With tiling held constant, smaller windows help under every rule — and help more than Otsu suggested.**

| Binarization rule | 32³/16 | 16³/8 | 12³/6 | 8³/4 | Trend |
|---|---|---|---|---|---|
| Otsu (published protocol) | 0.0972 | 0.1127 | 0.1169 | 0.1239 | +0.027 |
| top 10% | 0.1034 | 0.1046 | 0.1047 | 0.1048 | +0.001 |
| top 5% | 0.1774 | 0.1836 | 0.1855 | 0.1870 | +0.010 |
| **top 1% (deployable)** | 0.2968 | 0.3631 | **0.3693** | 0.3568 | **+0.060** |
| **oracle volume (diagnostic)** | 0.3085 | **0.4152** | 0.4079 | 0.3754 | **+0.067** |

RQ3b/RQ3c's central claim survives, and was *understated*: the benefit of a smaller query window is 2-3× larger under the calibrated rules (+0.060 pct99, +0.067 oracle) than under the Otsu protocol that discovered it (+0.027). The calibrated curves also reveal an interior optimum at **12³-16³** that Otsu's monotone curve hid, with a mild genuine decline by 8³.

![The 32³ baseline against an 8³ window at matched 50% overlap, on the same patients under the same Dice implementation. The advantage is real under every binarization rule and largest under the two that best reflect heatmap quality — the reverse of what the first, tiling-confounded version of this comparison showed.](figures/fig_rq12_threshold_reversal.png){width=98%}

**The 8³/6³ "continued improvement" in Section 7.4 was a tiling artifact, and it exposes Otsu directly.** Holding window size fixed at 8³ and changing only the stride:

| Binarization rule | 8³ overlapped | 8³ non-overlapped | Δ | p |
|---|---|---|---|---|
| Otsu | 0.1239 | 0.1407 | **+0.0169** (prefers non-overlapping) | 5.8×10⁻³³ |
| top 10% | 0.1048 | 0.1020 | −0.0028 | 2.1×10⁻³³ |
| top 5% | 0.1870 | 0.1738 | −0.0132 | 1.6×10⁻³⁵ |
| top 1% | 0.3568 | 0.2574 | **−0.0994** | 7.6×10⁻³³ |
| oracle volume | 0.3754 | 0.2652 | **−0.1101** | 3.5×10⁻²⁴ |

**Otsu and every calibrated rule disagree in *sign*, on the same heatmaps, by a wide margin.** Dropping overlap makes the heatmap blocky and low-entropy — a coarser intensity histogram, which gives Otsu's between-class-variance criterion a cleaner two-class split and therefore a *better* score, while every rule that selects a fixed fraction of voxels is penalised for the lost spatial resolution. This is the sharpest statement this project can make about its own instrument: **Otsu does not merely add noise, it rewards a specific degradation of the heatmap.** Section 7.4's reported "plateau at 6³ for ET and TC, while WT keeps improving" compares two non-overlapping points against three overlapping ones and is not a finding about window size at all.

![The window sweep re-scored under every binarization rule. Columns are window/stride; the first four hold the 50%-overlap convention fixed and the last two are the non-overlapping points the original sweep switched to. Only the Otsu curve keeps rising across that switch.](figures/fig_rq12_window_curve.png){width=98%}

**The pointing game confirms the corrected picture and localizes who benefits.** Comparing the 32³ baseline against 8³ at matched 50% overlap, using the tie-aware centroid rule:

| Region | Bin | 32³ hit rate | 8³/stride-4 hit rate | Change |
|---|---|---|---|---|
| ET | small | 0.000 | **0.190** | **+0.190** |
| ET | medium | 0.130 | **0.565** | **+0.435** |
| ET | large | 0.348 | **0.609** | **+0.261** |
| TC | small | 0.111 | 0.000 | −0.111 |
| TC | medium | 0.348 | 0.130 | −0.218 |
| TC | large | 0.696 | 0.217 | −0.479 |
| WT | small | 0.531 | **0.812** | **+0.281** |
| WT | medium | 0.857 | 0.857 | ±0.000 |
| WT | large | 0.850 | **1.000** | **+0.150** |

A smaller window improves pointing substantially for enhancing tumor and whole tumor — **including the bin that was at exact chance: small ET goes from 0 of 21 to 4 of 21** — while consistently degrading tumor core, the middle-sized region. That regional split is not something Dice showed at all, in either direction.

**What this section changes.** The claim that survives is stronger and more precise than the one Sections 7.3-7.4 made: a smaller query window genuinely improves localization, by more than the original protocol could detect, with an optimum around 12³-16³ rather than "as small as possible". What does not survive is the specific 8³/6³ extension and its plateau, both of which measured tiling rather than window size. And the methodological finding is the most transferable result here: a shared binarizer is *not* a cancelled confound, because arms can interact with it in opposite directions — which is exactly what happened, and what no amount of paired testing or FDR correction across those arms would have revealed.

### 7.12 RQ13: Do the retrained arms' verdicts survive a calibrated threshold?

Section 7.11 established that Otsu is not a neutral instrument on the *inference-side* arms. The retrained arms — RQ2, RQ4 and RQ6 — had still only ever been scored under it, and they are suspect for a specific mechanical reason: each builds its heatmap by a voxel-wise **max over three size-phrasing queries**, and a max over several maps changes the intensity histogram Otsu keys on. RQ2 maxes three same-scale maps; RQ4 and RQ6 max three *different-scale* maps. Neither resembles the baseline's single-query heatmap in histogram shape, so the comparison against that baseline was never guaranteed to be fair.

We re-scored all three under every binarization rule. The re-scored Otsu column reproduces each arm's published CSV to a pooled mean difference below 3.5×10⁻⁵ (93-98% of individual rows bit-identical; the remainder are the Otsu histogram-tie effect, which these ensemble arms hit more often than the single-query baseline does).

| Arm | Otsu | top 10% | top 5% | top 1% | oracle |
|---|---|---|---|---|---|
| **RQ2** (size-conditioned text) | **+0.023** *(better, p=8×10⁻¹⁷)* | −0.001 | −0.004 | +0.006 *(n.s.)* | +0.000 *(n.s.)* |
| RQ4 (scale-matched) | −0.038 | −0.007 | −0.024 | −0.032 | −0.024 |
| RQ6 (uniformity fix) | −0.023 | −0.008 | −0.036 | −0.060 | −0.040 |

**RQ2's improvement is the third Otsu artifact this project has found, and the most consequential.** Section 7.1 reported size-conditioned prompting as helping medium and large lesions substantially, at p=8×10⁻¹⁷. Under every calibrated rule that improvement disappears: +0.006 at top-1% (p=0.11, not significant), +0.000 under the oracle (p=0.67), and slightly negative at top-5% and top-10%. Per-region, whole tumor flips sign outright (+0.015 under Otsu, −0.033 under top-1%). The honest revision: **size-conditioned prompting does not improve localization; it produces a heatmap that Otsu happens to binarize more favourably.** Section 7.1's negative half — that it *worsens* small enhancing tumor and widens the ET size gap — is unaffected, and Section 7.10 already showed that half replicates across seeds.

**RQ4 and RQ6's negative verdicts are robust.** Both are worse than the baseline under all five rules, with the pooled magnitude actually larger under the calibrated ones. Whatever else is true of scale-matched retraining and the uniformity repair, they are not being unfairly penalised by the binarizer. (Per-region, tumor core flips to positive for both under top-1%, a reminder that pooled verdicts hide regional structure — but neither arm approaches the baseline overall.)

**Why this matters beyond these three arms.** Three of the four interventions this project evaluated on Dice have now had their Otsu verdict change under a calibrated threshold: RQ3b/RQ3c (understated by 2-3×), the 8³/6³ tiling points (an artifact entirely), and RQ2 (an artifact entirely). The one that did not change is the pair of clearly-negative arms. That pattern is itself informative — a miscalibrated binarizer distorts the ranking of *close* comparisons while leaving large effects intact, which is exactly the regime where careful ablation work lives.

### 7.13 Summary across ablations

\newpage

![Leaderboard: every method's mean Dice, one panel per region, bars grouped by size bin. These are Otsu-thresholded scores, the project's original protocol; Section 7.11 shows the absolute values here understate the smaller-window arms and that Otsu is not a neutral common factor across them.](figures/fig_leaderboard.png){width=98%}

**The bottom line, after Section 7.11.** One intervention works, and it is the simplest: **evaluating the frozen model at a smaller query window — around 12³-16³ rather than 32³ — with no retraining at all.** It is worth +0.060 Dice under the deployable top-1% rule and +0.067 under the oracle, roughly 2-3× more than the Otsu protocol that originally found it was able to show. It also improves threshold-free pointing accuracy for enhancing tumor in all three size bins, lifting the previously-at-chance ET-small bin from 0 of 21 patients to 4 of 21, at the cost of degrading tumor core.

Every other attempt failed to beat the plain baseline: text-side size conditioning (RQ2, whose apparent gain Section 7.12 shows was an Otsu artifact), naive multi-scale ensembling (RQ3), scale-matched retraining (RQ4, which a noise probe showed had learned resize artifacts), and the uniformity-regularizer repair (RQ6). Each produced genuine partial successes on its own terms; none surpassed doing nothing but changing the query window. Added complexity introduced new failure modes faster than it fixed the original one.

A second free improvement stands independently: **replacing Otsu with a fixed top-1% threshold**, worth 2-4× Dice (Section 6.3), needing no retraining and no ground truth.

**But the most transferable result is methodological, and it is a cautionary one.** Five conclusions in this report were overturned or materially qualified *after* being written up as findings — by the controls in Section 8 rather than by any significance test. Two of those reversals happened inside Section 7.11 alone, in opposite directions: the smaller-window win first appeared to be an Otsu artifact, then turned out to be real and *understated* once a confound between window size and tiling convention was removed. The general lesson is sharper than "check your metric": **a shared confound is not a cancelled confound.** Every arm in Section 7 used the same binarizer, which we argued made the comparisons internally fair. Section 7.11 shows Otsu and every calibrated rule disagree in *sign* about the same pair of heatmaps — so arms that interact differently with a shared instrument can be ranked backwards by it, and no amount of paired testing or FDR correction across those arms will reveal it.

## 8. Diagnostic Methodology

The ablation studies above rely on four recurring diagnostic tools, used consistently across every experiment rather than invented ad hoc per section, so that "significant" and "improved" mean what they claim to mean:

1. **Chance-level control** (Section 6.2): random-heatmap baseline run through the identical Otsu+Dice pipeline, to separate genuine model failure from Dice's known geometric bias toward large structures.
2. **Noise-probe diagnosis** (Sections 7.5-7.6): feeding pure synthetic noise through a trained model's real input pipeline to check whether an apparent capability (e.g. RQ4's improved classification accuracy) is grounded in real content or a shortcut/artifact.
3. **Cross-split replication** (Section 6.2): re-running the core finding on 2 additional independent random splits, not just reporting one lucky/unlucky split.
4. **Family-wise FDR correction** (Benjamini-Hochberg): every paired significance claim made anywhere in this report is corrected across the full accumulated family of tests, not evaluated against an uncorrected α=0.05.
5. **Pipeline decomposition** (Section 6.3): when a metric is produced by a multi-stage pipeline, measuring each stage's separate contribution rather than attributing the result to the stage of interest by default. Applied to the heatmap/threshold split, this showed that roughly half of the measured collapse for two of three regions came from the binarization step rather than from grounding — a confound invisible to the chance-level control in tool 1, because the control passes through the same binarization and the artifact cancels out of the ratio.
6. **Metric triangulation** (Section 6.3): confirming a finding with a second metric whose failure modes differ from the first. Dice conflates pointing with delineation and carries a geometric size penalty; the pointing game separates them and has no such penalty. Where the two agree the claim is robust; where they diverge — as at ET-small, at chance on pointing but merely "low" on Dice — the divergence itself is the finding.
7. **Replication at the right unit** (Sections 7.8, 7.10): a within-run paired test over 213 patients answers "is this difference real *for this trained model*", not "is this difference real". Since every arm here is one training run, the unit of replication is the run, and the honest evidence is sign consistency across independently seeded runs measured against a retraining noise floor. Applied to RQ7 this demoted a p≈6×10⁻³⁵ result to noise; applied to RQ5 it upgraded a reported null to a consistent positive.
8. **Metric well-definedness** (Section 7.11): checking that a metric measures what its name implies before interpreting it. The pointing game presumes a well-defined peak; a strided sliding window makes the heatmap piecewise-constant over blocks, so `argmax` returns a block corner and the "peak" is ambiguous by up to the stride. Measuring the size of the tied-maximum plateau exposed this and changed several Section 6.3 numbers substantially.
9. **Confound isolation** (Section 7.11): when two settings change together, no comparison between them means anything. Section 7.4's sweep shrank the window *and* switched from 50%-overlap to non-overlapping tiling at the 8³ point, which was noted at the time as a compute detail rather than treated as a confound. Adding the missing 8³/stride-4 control was what separated the two, and it reversed the reading twice.

**On the value of these tools.** Five claims in this report were overturned or materially qualified by tools 5-9 *after* they had already been written up as findings: the magnitude of the size collapse (tool 5), the RQ7 encoder effects and the RQ5 null (tool 7), the pointing-game hit rates (tool 8), the size and mechanism of the window-size benefit (tools 5 and 9 together), and RQ2's reported improvement, which tool 5 showed to be an artifact of the binarizer rather than a property of the model (Section 7.12). Each was plausible, statistically significant, and internally consistent.

The last of those is the most instructive because it moved twice. Re-scoring under better thresholds first suggested the window benefit was an Otsu artifact that reversed; adding the overlap control then showed the benefit is real, larger than Otsu could measure, and that what actually reversed was a tiling change masquerading as a window-size effect. A single control is not a verdict — the first correction was itself confounded. The general lesson: with a multi-stage pipeline and a shared metric, significance testing alone does not protect you, because every arm inherits the same confound and the comparison still looks clean.

Figure below visualizes the outcome of every one of those corrected significance tests at once — every ablation, every region, every size bin, in one panel:

![Every paired significance test run in this project, RQ1 baseline vs. each ablation, one row per comparison, one column per region×size bin. Blue = significantly better, red = significantly worse, white = no significant difference; solid color survives BH-FDR correction, pale color is significant only before correction.](figures/fig_significance_heatmap.png){width=98%}

Reading this figure end to end tells the whole story at a glance: RQ3b/RQ3c (rows 3-6) are almost entirely solid blue, the isolated-window fix genuinely working almost everywhere it's tested; RQ4 and RQ6 (rows 7-8) are almost entirely solid red, both training-side attempts genuinely underperforming the baseline; RQ5 (bottom row) is almost entirely white, as a robustness check finding no meaningful difference should look; and RQ2/RQ3 (rows 1-2) are a genuine mix, matching their more nuanced, region-dependent stories in Sections 7.1 and 7.2. See Section 7.13 for the substantive conclusion this figure supports.

Two caveats on reading it. This figure shows Otsu-thresholded comparisons: Section 7.11 shows that *understates* the RQ3b/RQ3c rows by 2-3×, so the blue there is real and conservative. And the RQ5 row's whiteness is a single-run result — Section 7.10 shows that across three seeds RQ5 is consistently better, not neutral.

## 9. Challenges

Several non-trivial obstacles came up during implementation, beyond the RQ2 negative result discussed above:

- **Kaggle API version mismatch.** A newly-generated Kaggle API access token turned out to require kaggle CLI ≥1.8.0, which in turn requires Python ≥3.11 — incompatible with the project's Python 3.10 environment. Resolved by using Kaggle's legacy API key format instead, which the older, compatible CLI version supports.

- **A silent dependency collision that broke CUDA.** Installing MONAI pulled in a much newer PyTorch build (with a different CUDA version, 13.0) than the one already validated as compatible with this cluster's GPUs (12.1). After pinning PyTorch back down, the code still failed with `libcudnn.so.9: cannot open shared object file`. The root cause: the `nvidia-*-cu12` and `nvidia-*-cu13` pip packages install their shared libraries into the *same* internal path (e.g. `nvidia/cudnn/lib/`), so uninstalling the newer package deleted the older package's actual `.so` files while leaving its package metadata claiming it was still intact. Fixed by force-reinstalling every affected `cu12` package to restore the real files, and pinning an older MONAI release that doesn't force a PyTorch upgrade in the first place.

- **A second, unrelated version conflict.** The `transformers` library refuses to load PubMedBERT's checkpoint format (an older `pytorch_model.bin`; no `safetensors` version exists in that model's repository) unless PyTorch ≥2.6, a defensive measure against a real PyTorch CVE. Bumping PyTorch would have risked reopening the CUDA compatibility problem above, so an older `transformers` release was pinned instead — a purely software-loading restriction, not a functional requirement.

- **A known BraTS2020 data quirk.** One patient's segmentation file uses a non-standard filename left over from the original hospital de-identification process (Section 4), which crashed the preprocessing script partway through a 369-patient run. This also exposed a design flaw in the first version of the preprocessing script — it only wrote its summary CSV once, at the very end, so the crash would have silently discarded already-computed results for the preceding patients. Fixed by writing results incrementally per patient and adding a filename fallback for the one irregular case.

- **No GPU on the login node.** The interactive machine has no GPU at all; a CPU-only correctness test confirmed the training code was *correct* (loss decreasing, no crashes) but took over 11 minutes of CPU time for a handful of tiny batches — impractical for real training. This required learning the cluster's SLURM job submission system (partitions, GPU resource requests, account/QOS) to actually run anything at meaningful scale, rather than treating it as optional infrastructure.

- **An architectural dead-end avoided before it was built.** Grad-CAM was the originally planned localization technique. On closer inspection, the trained model globally average-pools each patch down to a single embedding vector, meaning there is no spatial feature map left near the output for Grad-CAM to meaningfully back-propagate onto — it would have produced a near-single-voxel "heatmap." This was caught during design, before implementation, and a sliding-window similarity map was used instead, which matches how the model actually represents space.

- **BERT anisotropy.** Raw PubMedBERT sentence embeddings for the four class descriptions turned out to have ~0.99 pairwise cosine similarity to each other — meaning the frozen text encoder alone carries almost no discriminative signal, and the trainable projection head has to do essentially all the separating work during contrastive training. This was verified directly with a pairwise similarity check before committing to the training pipeline design.

## 10. Limitations

- **Single anatomy, single dataset.** All results are on BraTS2020 brain tumors. Whether the same size-dependent collapse and the same failure of size-conditioned prompting hold for other organs/lesion types (e.g. lung nodules, liver lesions) is untested here — Future Work below proposes this as the next check.
- **Modest per-bin sample sizes.** Size bins range from n=20 (WT large) to n=32 (WT small) patients. The overall monotonic pattern is consistent and large in effect size, but individual bin means, especially for ET (27/369 patients have no enhancing tumor at all, further shrinking that region's usable sample), should be read with that sample size in mind.
- **Lightweight backbone, not a competitive segmentation model.** A ResNet-10 with sliding-window Otsu thresholding is a deliberately simple architecture chosen to make the size-vs-quality relationship easy to isolate and interpret. Absolute Dice numbers from the text-conditioned arms should not be compared to state-of-the-art BraTS leaderboards. Section 6.1's P′ check bounds how much of that gap is the setup rather than a pipeline defect: the same data, split and metric under conventional dense supervision reach published-range Dice, so the low text-conditioned numbers are a property of the task formulation, not of broken plumbing.
- **Templated, not naturalistic, text.** Region descriptions are hand-written templates (Section 5), not real radiologist-authored report sentences. RQ5 tested one naturalistic rewrite; across three seeds it is consistently *better* than templated text rather than equivalent (Section 7.10), though not formally significant at n=3 runs. Either way it is one specific naturalistic phrasing, not a sample of real report variation.
- **Replication is 3 runs, not full k-fold.** RQ1's core pattern held in all 3 seeds (9/9 region×bin), and RQ2/RQ4/RQ5/RQ6 and all four RQ7 conditions have since been replicated across 3 seeds each (Sections 7.8, 7.10). This is a substantial strengthening over the single-run evidence, but 3 runs give the cross-seed *t*-tests almost no power, which is why those sections lead with sign consistency against a noise floor rather than with p-values. A full k-fold sweep would be stronger.
- **The Otsu threshold was not a neutral choice, and it changed a headline conclusion.** Otsu and the stride-16 window were chosen for transparency rather than to maximize Dice. Section 6.3 quantifies the price: 2.2-5.4× in Dice relative to an oracle-volume threshold, 6-223× volume over-prediction, and *anti*-correlation with true lesion size. An earlier draft argued the Section 7 comparisons were nonetheless internally fair because every arm shared the threshold. **That argument was wrong**, and Section 7.11 shows why: at a fixed window size, Otsu and every calibrated rule disagree in *sign* about which of two heatmaps is better (Otsu prefers the non-overlapping tiling by +0.017; the top-1% rule rejects it by −0.099). A shared confound is not a cancelled confound when arms interact with it differently. In this project that mattered in both directions — Otsu understated the smaller-window benefit while simultaneously making a tiling change look like a window-size improvement.
- **The oracle-volume threshold is a diagnostic, not a method.** It uses the ground-truth voxel count and is therefore unavailable at inference. It is reported only to bound how much of the collapse is attributable to thresholding. The `pct99` fixed-percentile rule is the deployable version, and it recovers much of the same benefit.
- **RQ6's fix is incomplete, and its diagnostic follow-up is not a deployable method.** The uniformity regularizer fixed 2 of 3 shortcut behaviors identified by the noise probe, not all 3 — a residual hub bias remains for the smallest-scale pipeline, and RQ6 still underperforms the plain RQ1 baseline in 7 of 9 bins. Separately, the single-scale oracle test used to isolate why ensembling helps RQ6 relies on the *true* size bin label, which is not available at real inference time — it is a mechanism-isolating diagnostic, not a proposed evaluation protocol.
- **The window-size optimum is bracketed, not pinned down.** Under calibrated thresholds the curve peaks somewhere in the 12³-16³ range and declines mildly by 8³ (Section 7.11), but we evaluated only 32/16/12/8 at matched overlap, so the true optimum and the shape around it are uncharacterised. The plateau reported in Section 7.4 has been withdrawn: it rested on the 6³ and 8³ points, which changed the tiling convention as well as the window size.

## 11. Future Work

- **A smarter cross-scale combination rule.** RQ3 and RQ6 together show the failure mode isn't multi-scale representation learning itself, nor is ensembling inherently harmful (RQ6 benefits from it) — but naive voxel-wise max lets the noisiest scale win when the underlying model wasn't trained to be calibrated at that scale (RQ3). Worth trying: a learned gating/attention mechanism across scales, or weighting by each scale's calibration/confidence rather than taking a raw max.
- **Finish fixing the residual hub bias from RQ6.** The uniformity regularizer fixed 2 of 3 noise-probe shortcut behaviors and improved localization over RQ4 in all 9 bins, but the smallest-scale pipeline still shows residual bias toward the "large" class, and RQ6 still trails the plain RQ1 baseline overall. Worth trying: hard-negative mining that specifically contrasts the "large" and "small" classes during training, or a stronger uniformity weight, to see if the last residual bias can be closed and RQ6 can be pushed past the RQ1 baseline rather than just past RQ4.
- **Close the gap between P′ and the text-conditioned model.** Section 6.1 shows a supervised U-Net reaches 0.64 Dice on the smallest enhancing tumors where the text-conditioned model reaches 0.01, on identical data. That 60× gap is the real target, and it is now bounded rather than speculative. The obvious intermediate is a text-conditioned model with a dense decoder rather than a globally-pooled encoder, which would remove the sliding-window bottleneck entirely and make Grad-CAM-style attribution viable again.
- **Make the text pathway carry information at all.** RQ7 and RQ8 together show the query is a class index. Worth trying: contrastive negatives built from *within*-region text perturbations (true description vs. its negation as a hard negative pair), which would force the projection head to separate polarity rather than only region identity — the single cheapest change that could make the language side do work.
- **Pin down the window-size optimum.** Section 7.11 brackets it at 12³-16³ under calibrated thresholds but evaluates only four overlap-matched points. A denser sweep (24³, 20³, 14³, 10³) at fixed 50% overlap would locate it properly, and is cheap — no retraining, just re-querying the frozen model.
- **Extend to a second dataset** (e.g. LIDC-IDRI lung nodules) to test whether the failure pattern generalizes beyond brain tumors.

## 12. Effort / Contribution

This was an individual project; all design decisions, code, experiments, and writing below were done solo.

**What I had to learn.** Going in, I had not worked with MONAI, 3D medical image formats (NIfTI, multi-modal MRI co-registration), or SLURM job scheduling — all three were new to me and required real ramp-up: understanding how MONAI's 3D ResNet expects channel/volume layout, how to normalize and resample NIfTI volumes correctly (z-scoring within a brain mask rather than globally, which matters for skull-stripped data), and how to structure `sbatch` scripts against this cluster's specific partitions (`dev` for smoke tests capped at 10 minutes, `general` for real training/eval runs, correct account and GPU resource flags). I also had not previously implemented contrastive text-image (here, text-volume) alignment from scratch, so getting the shared embedding space, temperature-scaled cosine similarity loss, and sliding-window inference-time localization working correctly took real iteration — including recognizing partway through that Grad-CAM would not work at all on a globally-pooled architecture, before wasting time implementing it. On the statistics side, I had used t-tests before but not paired Wilcoxon signed-rank tests or Benjamini-Hochberg FDR correction across an accumulating family of tests, both of which turned out to be essential once the project grew past a single comparison — without FDR correction, several of the "significant" results in Section 7 would not have been defensible.

**What I already knew.** I came in comfortable with Python and PyTorch — writing custom `Dataset`/`DataLoader` classes, training loops, optimizers and schedulers, checkpointing, and reading loss curves to tell a bug apart from a bad hyperparameter — so none of the mechanics of building and training a model was new. I had a working understanding of transformer language models and of BERT specifically (tokenization, mean-pooling versus CLS extraction, using a frozen encoder as a feature extractor), which is why the anisotropy problem in Section 5 was something I thought to check for rather than something that ambushed me. I was already fluent with numpy and pandas for data wrangling, with matplotlib for figures, and with git. On the statistics side I was comfortable with correlation, t-tests, and the general logic of null-hypothesis testing, though not with the non-parametric and multiple-comparison machinery this project ended up needing. I also had prior exposure to CNNs and 2D computer vision, which transferred partially — the concepts carried over, but essentially every practical detail of working in 3D (memory budgeting, patch sampling, anisotropic voxel spacing) did not.

**Rough time split.** Approximately: 8% reading related work (the 3D medical VLM grounding-failure literature in Section 3, plus the methods papers cited in Section 5.1); 15% environment and data setup (BraTS2020 download, preprocessing, and — unexpectedly costly — resolving the CUDA/PyTorch/transformers dependency conflicts in Section 9); 20% writing the core pipeline (text encoder, volume encoder, contrastive training loop, sliding-window localization); 12% debugging (mostly the dependency issues, plus the incremental-CSV-write bug caught before it cost a full re-run); 25% designing and running the experiment sequence (RQ1 through RQ12, including the noise probes, the P′ supervised reference, and the seed replications); 8% on the statistics layer specifically (paired testing, FDR correction across an accumulating family, bootstrap intervals, and reworking the analysis once I realized the unit of replication was the training run and not the patient); and 12% writing and revising this report.

**Where the time actually went, versus where I expected.** I expected the bulk of the effort to be in getting a model to work. It was not — the baseline trained on essentially the first serious attempt. The real cost was in *checking* results, and the checks were where the project's actual content came from. Four conclusions I had already written up as findings did not survive their own follow-up test (Section 8), and the last of them — discovering that this project's apparent best intervention won only under the threshold I had standardized on — arrived late enough that it required rewriting the report's conclusion rather than just adding a caveat. If I ran this project again I would build the diagnostic layer first and the model second.

## References

**Models and representation learning**

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. In *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139:8748-8763.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. *arXiv:1807.03748*. (Origin of the InfoNCE objective used in Section 5.1.)
- Wang, T., & Isola, P. (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. In *Proceedings of the 37th International Conference on Machine Learning (ICML)*, PMLR 119:9929-9939. (Basis of the RQ6 uniformity regularizer.)
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 3982-3992.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778. (Architecture underlying the MONAI 3D ResNet-10 volume encoder.)
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 234-241. (Architecture of the P′ supervised reference in Section 6.1.)

**Methods: thresholding, metrics, and statistics**

- Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62-66.
- Zhang, J., Bargal, S. A., Lin, Z., Brandt, J., Shen, X., & Sclaroff, S. (2018). Top-down Neural Attention by Excitation Backprop. *International Journal of Computer Vision*, 126, 1084-1102. (Source of the pointing-game localization metric used in Sections 6.3 and 7.9.)
- Dice, L. R. (1945). Measures of the Amount of Ecologic Association Between Species. *Ecology*, 26(3), 297-302.
- Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6), 80-83.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- Kerby, D. S. (2014). The Simple Difference Formula: An Approach to Teaching Nonparametric Correlation. *Comprehensive Psychology*, 3, 11.IT.3.1. (Matched-pairs rank-biserial effect size.)
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. (Percentile bootstrap confidence intervals.)

**Application domain**

- Chen, X., Shi, B., Le, C., Yin, Q., Lin, L., Ni, H., Gong, R., & Li, P. (2026). Auditing Frontier Vision-Language Models for Trustworthy Medical VQA: Grounding Failures, Format Collapse, and Domain Adaptation. *arXiv:2604.27720*.
- Chen, Y., Xiao, W., Bassi, P. R. A. S., Zhou, X., Er, S., Hamamci, I. E., Zhou, Z., & Yuille, A. (2025). Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering. *arXiv:2505.18915*.
- Koleilat, T., Asgariandehkordi, H., Rivaz, H., & Xiao, Y. (2024). MedCLIP-SAMv2: Towards Universal Text-Driven Medical Image Segmentation. *arXiv:2409.19483*. (Published in *Medical Image Analysis*, 2025.)
- Xie, Y., Zhou, J., Wang, R., Zhang, J., & Xia, Y. (2024). SimTxtSeg: Weakly-Supervised Medical Image Segmentation with Simple Text Cues. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*.
- Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS). *IEEE Transactions on Medical Imaging*, 34(10), 1993-2024.
- Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., & Poon, H. (2021). Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing. *ACM Transactions on Computing for Healthcare*, 3(1), 1-23. (PubMedBERT; weights at `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`.)
- MONAI Consortium. MONAI: Medical Open Network for AI. https://monai.io
