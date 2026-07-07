# Architecture, Not Language: Diagnosing Small-Lesion Grounding Failure in Text-Conditioned 3D Medical Localization

MPCS 53113 Natural Language Processing mini-project, University of Chicago.

## Problem

Text-conditioned 3D medical models can detect that a lesion exists but often fail to localize it precisely when it's small, collapsing to oversized and imprecise regions instead. This project quantifies that failure on BraTS2020 brain MRI, then tests four candidate fixes (size-conditioned prompting, multi-scale windowing, scale-matched retraining, and naturalistic text) with statistically corrected paired testing throughout.

## Methodology

PubMedBERT text embeddings (mean-pooled region descriptions) are aligned with 3D ResNet-10 patch embeddings (MONAI) in a shared contrastive space, trained on 32³ patches sampled from BraTS2020 volumes (128³, z-score normalized) against four classes: enhancing tumor, tumor core, whole tumor, and normal tissue. Since the trained model globally pools each patch to a single embedding, Grad-CAM isn't viable, so localization is done via a sliding-window similarity map, sweeping the trained encoder across the full volume and thresholding with Otsu's method to get a predicted mask. Evaluation stratifies held-out patients into small/medium/large tercile bins per lesion volume (computed at native resolution, not the resampled grid) and reports Dice/IoU against ground truth, alongside a chance-level random-heatmap control to rule out Dice's inherent size bias. Follow-up experiments are evaluated against this same protocol using paired Wilcoxon tests with Benjamini-Hochberg correction across the full family of comparisons.

## Findings

The contrastive alignment baseline learns genuine signal (validation accuracy 0.671 vs. 0.25 chance). The core result: localization quality collapses monotonically with lesion size across all three tumor subregions (5 to 15x Dice degradation from large to small), surviving a chance-level control, and replicating exactly (9 of 9 region×bin comparisons) across two additional independent train/val splits. Size-conditioned prompting improves medium/large lesions but significantly worsens small enhancing-tumor localization. Isolating a single smaller sliding window improves tumor core and whole tumor localization after correcting for multiple comparisons; pushing the window size down further (12³, then 8³) shows the improvement continuing with no plateau, and the enhancing-tumor region's statistical picture actually strengthens at smaller windows, clearing full-family-corrected significance at 8³. Retraining with scale-matched patches reaches better classification accuracy but produces significantly worse localization everywhere; a pure-noise probe confirms the model keys off resize-interpolation artifacts rather than genuine tumor content, a directly-verified case of shortcut learning. Replacing templated text with naturalistic radiology-style language reproduces the original finding almost exactly, ruling out template phrasing as a confound.

Full writeup: [`report_draft.pdf`](report_draft.pdf) / [`report_draft.md`](report_draft.md).

## Repository layout

- `src/` — all code. See [`src/README.md`](src/README.md) for a file-by-file breakdown.
- `slurm/` — SLURM batch scripts used to run training/evaluation on the cluster's GPU partitions.
- `results/` — per-patient CSV outputs (Dice/IoU scores) from every evaluation run.
- `figures/` — generated report figures.
- `report_draft.md` / `report_draft.pdf` — the final report.

Not included in this repo (see below for how to regenerate):
- `data/` — BraTS2020 raw + preprocessed volumes (~12GB).
- `checkpoints/` — trained model weights (~400MB).
- `logs/` — raw SLURM job stdout.

## Reproducing this project

1. **Environment**: Python 3.10, PyTorch 2.4.0+cu121, MONAI 1.4.0, transformers 4.40.0. See `src/` imports for the full dependency list.
2. **Data**: download BraTS2020 from Kaggle (`awsaf49/brats20-dataset-training-validation`) into `data/BraTS2020_TrainingData/`.
3. **Preprocess**: `python src/preprocess.py` — normalizes volumes, resizes to 128³, computes per-patient lesion volumes for size-bin stratification.
4. **Text embeddings**: `python src/text_encoder.py` — embeds all region descriptions (base, size-conditioned, and naturalistic variants) with PubMedBERT.
5. **Train the P′ baseline**: `python src/train_baseline.py` (see `slurm/train_baseline.sbatch` for the cluster job).
6. **Run RQ1 through RQ5**: see the corresponding `train_rqN.py` / `evaluate_rqN.py` scripts and their matching `slurm/*.sbatch` files.
7. **Statistical analysis**: `python src/analyze_all_comparisons.py` reproduces every paired significance test with BH-FDR correction.
8. **Figures**: `python src/make_figures.py`, `python src/make_figure4_scale_comparison.py`, `python src/make_overlay_figure.py`.

## Attribution

All code in this repository was written for this project. Third-party dependencies (PyTorch, MONAI, Hugging Face Transformers, PubMedBERT weights) are used via their public APIs/package installs, not vendored.
