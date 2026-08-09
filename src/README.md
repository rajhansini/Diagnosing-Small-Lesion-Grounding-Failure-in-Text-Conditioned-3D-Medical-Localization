# `src/` — file guide

**Attribution.** Every file below was written by me for this project. None contains code copied from another project, another assignment, or a third-party codebase. Third-party libraries (PyTorch, MONAI, Hugging Face Transformers, scikit-image, SciPy, matplotlib) and the pretrained PubMedBERT weights are used through their public APIs and package installs; nothing is vendored into this repository.

**Totals: 60 files, 10,323 lines.** Line counts below are `wc -l` including docstrings and comments.

Run order for a full reproduction is documented in the [root README](../README.md).

---

## Data pipeline

| File | LOC | Purpose |
|---|---|---|
| `preprocess.py` | 190 | Loads raw BraTS2020 NIfTI volumes, z-score normalizes per modality *within the brain mask*, resizes to 128³, and computes per-patient/per-region lesion volumes at **native** resolution for size-bin stratification. Handles the `BraTS20_Training_355` misnamed-segmentation quirk and writes results incrementally so a mid-run crash cannot discard completed work. |
| `text_encoder.py` | 204 | PubMedBERT wrapper and the single source of all text used in the project. Defines and embeds the base region descriptions (ET/TC/WT/NONE), the size-conditioned variants (RQ2/RQ4), and the naturalistic radiology-style variants (RQ5). |
| `build_rq7_text_variants.py` | 166 | Builds the four RQ7 text-encoder ablation conditions: BERT-base, random-init BERT, random orthonormal vectors, and random vectors resampled to PubMedBERT's anisotropic geometry. |
| `build_rq8_probe_texts.py` | 181 | Builds the five RQ8 compositionality probes (original / negated / shuffled / swapped / generic) and asserts the `original` embedding matches the training embedding to floating-point tolerance. |

## Datasets and model

| File | LOC | Purpose |
|---|---|---|
| `dataset.py` | 102 | Patch dataset for the baseline / RQ1 / RQ5 / RQ7: one labeled 32³ patch per (patient, region). Defines `region_mask()`, the single definition of ET/TC/WT used everywhere in the project. |
| `dataset_rq2.py` | 110 | Size-conditioned dataset: labels each patch with its region's true size bin for the 10-way task. Physical crop size is held fixed, so only the *text* varies. |
| `dataset_rq4.py` | 90 | Scale-matched dataset: crop size tied to the true size bin (16³/32³/64³), resized to the canonical 32³ input. Changes the geometry as well as the text. |
| `dataset_pprime.py` | 105 | Full-volume dense-label dataset for the P′ supervised reference, with tumor-biased random cropping. |
| `model.py` | 51 | `TextVolumeAligner`: MONAI 3D ResNet-10 volume encoder + linear text projection into a shared 256-d L2-normalized space. |
| `localize.py` | 72 | `sliding_window_heatmap()`: sweeps the encoder across a volume accumulating per-voxel cosine similarity to a text query. Supports querying at a different physical window size than the model was trained at, which is what makes the entire window sweep possible without retraining. |

## Training

| File | LOC | Purpose |
|---|---|---|
| `train_baseline.py` | 123 | Contrastive alignment baseline (RQ1). 4-way ET/TC/WT/NONE classification. `--seed` controls the train/val split for cross-seed replication. |
| `train_rq2.py` | 128 | RQ2: 10-way size-conditioned classification at a fixed patch size. |
| `train_rq4.py` | 128 | RQ4: 10-way size-conditioned classification with scale-matched patch sampling. |
| `train_rq5.py` | 128 | RQ5: baseline trained against naturalistic rather than templated text. |
| `train_rq6.py` | 146 | RQ6: RQ4's setup plus a uniformity regularizer penalizing high pairwise cosine similarity among projected class embeddings, to break the "large"-class hub RQ4 diagnosed. |
| `train_rq7.py` | 152 | RQ7: baseline trained against one substituted text-encoder condition. |
| `train_pprime_supervised.py` | 195 | **P′ reference point.** Conventional supervised 3D U-Net segmentation on the identical split, data and metric — the previously-studied problem used to validate that the shared pipeline is sound. |

## Evaluation

| File | LOC | Purpose |
|---|---|---|
| `evaluate_rq1.py` | 185 | Core size-stratified Dice/IoU evaluation. Defines `otsu_threshold()`, `dice_iou()` and `size_bin()`, which **every** other evaluation script imports rather than redefining, so all arms share one metric implementation. |
| `evaluate_rq2.py` | 121 | RQ2 with the size-phrasing ensemble (max over small/medium/large queries). |
| `evaluate_rq3_multiscale.py` | 113 | RQ3: multi-scale window ensemble (16³/32³/64³, voxel-wise max) on the frozen baseline. |
| `evaluate_rq3b_scale16_only.py` | 73 | RQ3b: the 16³ window alone, isolating the receptive-field effect from the ensembling effect. |
| `evaluate_window_sweep.py` | 86 | RQ3c: any single (window, stride) pair via CLI. Produced the 12³, 8³ and 6³ points. |
| `evaluate_rq4.py` | 126 | RQ4 with scale-matched window querying per size phrasing. |
| `evaluate_rq5.py` | 112 | Size-stratified evaluation of the naturalistic-text model. |
| `evaluate_rq6.py` | 124 | RQ6 under RQ4's protocol, for a direct paired comparison. |
| `evaluate_rq6_single_scale.py` | 113 | Oracle test isolating whether RQ6's benefit comes from the embedding fix or from cross-scale ensembling. |
| `evaluate_rq7.py` | 119 | One text-encoder ablation under the identical RQ1 protocol. |
| `evaluate_rq8_compositionality.py` | 193 | RQ8 probes. Carries a built-in correctness gate: its `original` condition must reproduce `rq1_localization_scores.csv` exactly. |
| `evaluate_rq11_threshold_confound.py` | 234 | RQ11: recomputes each heatmap once and scores it under five binarization rules plus the pointing game, decomposing the collapse into grounding and thresholding. |
| `evaluate_grounding_sweep.py` | 200 | RQ12: generalizes RQ11 to any window size, with a **tie-aware** pointing metric. At window 32 it must reproduce RQ11's CSV — an end-to-end correctness gate. |
| `evaluate_pprime_supervised.py` | 118 | Scores the P′ segmenter against published BraTS Dice ranges and by the project's own size terciles. |
| `evaluate_ablation_thresholds.py` | 197 | RQ13: re-scores the retrained arms (RQ2/RQ4/RQ6) under all five binarizers, reproducing each arm's published ensemble heatmap construction exactly. |

## Diagnostics and controls

| File | LOC | Purpose |
|---|---|---|
| `compute_chance_baseline.py` | 60 | Random-heatmap control through the identical Otsu+Dice pipeline, separating model failure from Dice's geometric bias against small structures. |
| `sanity_check_localize.py` | 72 | Pre-flight check that the heatmap scores higher inside the true region than outside, before the localizer is trusted for evaluation. |
| `test_rq4_shortcut_hypothesis.py` | 102 | Feeds pure noise through RQ4's three resize pipelines to test directly whether the model learned resize artifacts rather than tumor content. |
| `test_rq6_hub_bias.py` | 75 | Re-runs that noise probe on the RQ6 checkpoint to verify the uniformity regularizer actually broke the embedding hub. |

## Analysis

| File | LOC | Purpose |
|---|---|---|
| `analyze_full_family.py` | 267 | The project's central statistics script: recomputes all 171 paired Wilcoxon tests from the saved CSVs and applies Benjamini-Hochberg correction, under both a pooled family and a per-research-question family. |
| `analyze_seed_replication.py` | 209 | Tests whether the RQ2/RQ4/RQ5/RQ6 conclusions hold across three independent training runs, judging by sign consistency against a retraining noise floor rather than within-run p-values. |
| `analyze_rq7.py` | 251 | RQ7 single-seed analysis, including an equivalence test and the semantics-vs-geometry decomposition. |
| `analyze_rq7_multiseed.py` | 180 | RQ7 across three seeds. Exposes the pseudo-replication in the single-seed p-values. |
| `analyze_rq8.py` | 184 | RQ8: how far each manipulation moves the query versus how far it moves behaviour. |
| `analyze_rq11.py` | 147 | RQ11: threshold calibration, the cost of the Otsu step, and the pointing game vs. chance. |
| `analyze_rq12.py` | 272 | RQ12: the tie artifact, the pointing comparison across window sizes, the overlap contrast, and whether the smaller window's Dice win survives a better binarizer. |
| `analyze_rq13.py` | 141 | RQ13: whether the retrained arms' Section 7 verdicts survive a calibrated threshold, with a reproduction gate on the Otsu column. |
| `analyze_appendix.py` | 563 | The eight statistics blocks the result tables held and no other script had extracted: IoU alongside Dice, effect sizes and bootstrap intervals across the family, the two BH correction schemes compared, checkpoint-selection sensitivity, the three pointing rules with a block-level chance baseline computed from the masks, what each binarizer actually selects across the sweep, per-region window optima, and the cost of the recommended intervention. Backs Appendix A of the report. |
| `analyze_all_comparisons.py` | 109 | Earlier consolidated statistics script, superseded by `analyze_full_family.py` but retained because Sections 7.1–7.6 were first computed with it. |
| `analyze_results.py` | 71 | Early-stage stats: Spearman correlation and lift over chance for RQ1. |

## Figures

| File | LOC | Purpose |
|---|---|---|
| `make_figures.py` | 107 | Figure 1 (Dice vs. volume scatter with chance overlay) and Figure 2 (RQ1 vs. RQ2 bars). |
| `make_overlay_figure.py` | 67 | Qualitative heatmap overlays on example large/small lesion slices. |
| `generate_example_heatmaps.py` | 51 | Recomputes and caches the two heatmaps the overlay figure uses. |
| `make_figure4_scale_comparison.py` | 71 | RQ1 vs. RQ3b vs. RQ4 grouped bars. |
| `make_figure5_window_curve.py` | 68 | Mean Dice vs. window size (32³→6³), one line per region. |
| `make_figure_architecture.py` | 106 | Pipeline schematic for the Method section. |
| `make_figure_leaderboard.py` | 70 | Leaderboard bars: every method's mean Dice, one panel per region. |
| `make_figure_roadmap.py` | 141 | Decision-tree diagram of every ablation, colour-coded by outcome. |
| `make_figure_significance_heatmap.py` | 105 | Every paired significance test as one heatmap, recomputing the tests from the CSVs. |
| `make_figures_dataset_and_evidence.py` | 466 | The two dataset figures (split composition, per-region volume distributions with tercile cutoffs) plus four evidence figures for claims that were previously tables only: Otsu's inverse calibration, the pointing game against chance, per-seed ablation replication, and the RQ4 shortcut noise probe. |
| `make_figures_rq7_rq8_rq12.py` | 329 | The four late-experiment figures: the RQ12 threshold reversal, RQ7 per-seed deltas against the retraining noise floor, RQ8 heatmap correlations under query manipulation, and the P′ supervised-vs-text-conditioned size comparison. |
| `make_figures_supplementary.py` | 885 | Nine further evidence figures, one per claim the report had previously stated only as a number: PubMedBERT's anisotropy before and after the trained projection (with every RQ7 condition placed on that axis), the chance-baseline lift, the five-binarizer threshold ladder, peak-to-lesion distance distributions, RQ3's naive multi-scale losses, classification accuracy against localization quality across all five training arms, RQ6's three-part uniformity verification, RQ8's embedding-displacement scatter, and RQ13's re-scoring of the retrained arms. Reads the training logs as well as the result CSVs. |
| `make_figures_appendix.py` | 769 | Nine figures for measurements that were computed and stored by the original runs but never looked at: Dice against IoU, the per-patient distributions behind every mean, all 171 tests by effect size against adjusted significance, the Benjamini-Hochberg step-up and a growing-family check, RQ7 at both saved checkpoints, the three pointing rules against a block-level chance baseline, what fraction of the volume each binarizer selects across the sweep, the window curve per region, and forward-pass cost against Dice. Reads the preprocessed masks as well as the result CSVs. |

---

## Conventions worth knowing

1. **Metrics are imported, never redefined.** `otsu_threshold()`, `dice_iou()` and `size_bin()` live in `evaluate_rq1.py` and are imported by every other evaluation script — including the supervised P′ — so cross-experiment comparisons cannot be confounded by differing implementations.
2. **Splits are reproducible from a seed alone.** Every training and evaluation script derives its train/val split from `sorted(listdir())` plus a seeded shuffle, so passing the same `--seed` anywhere reconstructs the same held-out patients without storing a split file.
3. **Several scripts carry built-in correctness gates.** `evaluate_rq8_compositionality.py` asserts its control condition reproduces RQ1's CSV; `evaluate_grounding_sweep.py` at window 32 must reproduce RQ11's; `evaluate_ablation_thresholds.py` requires its re-scored Otsu column to reproduce each arm's published CSV. Two of the three caught real bugs during development.
4. **Every number in the report is recomputed from the per-patient CSVs in `results/`** by a script in this directory. No statistic is transcribed by hand.
