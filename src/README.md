# src/ — file guide

All files below were written by me for this project. None contain code copied from another project or assignment.

## Data pipeline

| File | LOC | Purpose |
|---|---|---|
| `preprocess.py` | 124 | Loads raw BraTS2020 NIfTI volumes, z-score normalizes per modality, resizes to 128³, computes per-patient/per-region lesion volumes (native resolution) for size-bin stratification. Handles the `BraTS20_Training_355` misnamed-segmentation-file quirk. |
| `text_encoder.py` | 182 | PubMedBERT wrapper. Defines and embeds three text variants: base region descriptions (ET/TC/WT/NONE), size-conditioned descriptions (RQ2/RQ4), and naturalistic radiology-style descriptions (RQ5). |

## Datasets and model

| File | LOC | Purpose |
|---|---|---|
| `dataset.py` | 57 | Patch dataset for the P′ baseline / RQ1 / RQ5: samples one labeled 32³ patch per (patient, region) pair. |
| `dataset_rq2.py` | 70 | Size-conditioned patch dataset: labels each patch with its region's true size bin for the 10-way classification task. |
| `dataset_rq4.py` | 64 | Scale-matched patch dataset: crop size tied to the true size bin (16³/32³/64³), resized to the model's canonical 32³ input. |
| `model.py` | 17 | `TextVolumeAligner`: MONAI 3D ResNet-10 volume encoder + linear text projection, contrastive alignment. |
| `localize.py` | 45 | Sliding-window similarity map. Supports querying at a different physical window size than the model's input size (resized via trilinear interpolation), used for the RQ3/RQ3b/RQ4 multi-scale experiments. |

## Training

| File | LOC | Purpose |
|---|---|---|
| `train_baseline.py` | 91 | P′ baseline / RQ1: 4-way (ET/TC/WT/NONE) contrastive classification. Takes a `--seed` argument controlling the train/val split, used for the cross-validation check across seeds 0/1/2. |
| `train_rq2.py` | 94 | RQ2: 10-way size-conditioned contrastive classification, fixed 32³ patches. |
| `train_rq4.py` | 94 | RQ4: 10-way size-conditioned classification with scale-matched patch sampling. |
| `train_rq5.py` | 94 | RQ5: same as baseline, trained against naturalistic-text embeddings instead of templated. |

## Evaluation

| File | LOC | Purpose |
|---|---|---|
| `evaluate_rq1.py` | 135 | Core size-stratified Dice/IoU evaluation. Defines `otsu_threshold`, `dice_iou`, `size_bin` used throughout the rest of the project. Takes a `--seed` argument matching `train_baseline.py`'s, for evaluating the seed 0/1/2 cross-validation checkpoints. |
| `evaluate_rq2.py` | 101 | Evaluates RQ2 with the size-phrasing ensemble (max over small/medium/large text queries). |
| `evaluate_rq3_multiscale.py` | 104 | RQ3: multi-scale window ensemble (16³/32³/64³, max combination) on the frozen RQ1 model. |
| `evaluate_rq3b_scale16_only.py` | 73 | RQ3b: isolates the 16³ window alone (no ensembling) to separate the receptive-field effect from the ensembling effect. |
| `evaluate_window_sweep.py` | 85 | RQ3c: generalizes RQ3b to any single (window_size, stride) pair via CLI args. Used for the 12³ and 8³ points extending the window-size curve. |
| `evaluate_rq4.py` | 106 | Evaluates RQ4 with scale-matched window querying per size-phrasing. |
| `evaluate_rq5.py` | 92 | Size-stratified evaluation for the naturalistic-text model. |
| `compute_chance_baseline.py` | 55 | Random-noise-heatmap control run through the identical Otsu+Dice pipeline, on the same patients/regions as RQ1. |
| `sanity_check_localize.py` | 71 | Verifies the sliding-window heatmap scores higher inside the true region than outside, before trusting it for evaluation. |
| `test_rq4_shortcut_hypothesis.py` | 96 | Feeds pure random noise through RQ4's three resize pipelines and checks whether the model still assigns systematically different scores, i.e. whether it learned resize artifacts rather than content. |

## Analysis and figures

| File | LOC | Purpose |
|---|---|---|
| `analyze_results.py` | 62 | Early-stage stats: Spearman correlation and lift-over-chance for RQ1. |
| `analyze_all_comparisons.py` | 90 | Consolidated statistics: recomputes every paired Wilcoxon test (RQ1 vs. RQ2/RQ3/RQ3b/RQ4/RQ5) from the saved CSVs and applies Benjamini-Hochberg FDR correction across the full family. |
| `make_figures.py` | 97 | Figure 1 (Dice vs. volume scatter with chance overlay) and Figure 2 (RQ1 vs. RQ2 bar chart). |
| `make_overlay_figure.py` | 58 | Figure 3: qualitative heatmap overlay on example large/small lesion slices. |
| `make_figure4_scale_comparison.py` | 62 | Figure: RQ1 vs. RQ3b vs. RQ4 grouped bar chart. |
| `make_figure5_window_curve.py` | 57 | Figure: mean Dice vs. window size (32³/16³/12³/8³), one line per region, for the RQ3c curve. |
| `generate_example_heatmaps.py` | 50 | Recomputes and saves the two example heatmaps used by `make_overlay_figure.py`. |

**Total: 2,326 lines across 26 files.**
