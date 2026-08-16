# Architecture, Not Language: Diagnosing Small-Lesion Grounding Failure in Text-Conditioned 3D Medical Localization

MPCS 53113 Natural Language Processing mini-project, University of Chicago.

## Problem

Text-conditioned 3D medical models can detect that a lesion exists but often fail to localize it precisely when it's small, collapsing to oversized and imprecise regions instead. This project quantifies that failure on BraTS2020 brain MRI, then asks three questions about it: **is it real** (validated against a supervised reference and a chance control), **is it linguistic** (text-encoder ablations and compositionality probes), and **can it be fixed** (five candidate interventions) — with statistically corrected paired testing throughout, and with each conclusion re-checked by a control designed to break it.

## Methodology

PubMedBERT text embeddings (mean-pooled region descriptions) are aligned with 3D ResNet-10 patch embeddings (MONAI) in a shared contrastive space, trained on 32³ patches sampled from BraTS2020 volumes (128³, z-score normalized) against four classes: enhancing tumor, tumor core, whole tumor, and normal tissue. Since the trained model globally pools each patch to a single embedding, Grad-CAM isn't viable, so localization is done via a sliding-window similarity map, sweeping the trained encoder across the full volume and thresholding to get a predicted mask. Evaluation stratifies held-out patients into small/medium/large tercile bins per lesion volume (computed at native resolution, not the resampled grid) and reports Dice/IoU against ground truth.

Four measurement controls sit underneath every claim: a **chance-level random-heatmap baseline**, a **P′ supervised reference** (a conventional U-Net on the identical data, split and metric, checked against published BraTS Dice), **five binarization rules** rather than one so threshold effects can be separated from grounding, and a **threshold-free pointing game**. Every comparison is a paired Wilcoxon test with Benjamini-Hochberg correction across the full accumulated family of 171 tests, and every retrained arm is replicated across three independent seeds — because the unit of replication is the training run, not the patient.

## Findings

**The failure is real, and specific to text conditioning.** Localization collapses monotonically with lesion size across all three tumor subregions — 2.4–14.4x Dice degradation large-to-small once binarization is controlled for, and 5–15x as first measured under the unsupervised Otsu threshold — survives a chance-level control, and replicates in 9 of 9 region×bin comparisons across three independent splits. A **P′ check** — a conventional supervised U-Net trained on the identical data, split and Dice implementation — reaches published BraTS Dice (ET 0.758 / TC 0.812 / WT 0.851) and degrades by only 1.2–1.3x large-to-small. Small lesions are therefore learnable on this data, and Dice's geometric size penalty does not explain the collapse; text-conditioned grounding is what fails.

**About half the measured collapse was a metric artifact.** The unsupervised Otsu threshold over-predicts lesion volume by 6–223x and is *anti*-correlated with true lesion size (ρ = −0.26 to −0.48). Under an oracle-volume threshold the large/small ratio falls from 15.0/9.2/4.9 to 14.4/5.6/2.4. A threshold-free pointing game locates what remains: roughly uniform 46–122x chance accuracy everywhere, except small enhancing tumor, where the peak response lands inside the true lesion in **0 of 21 patients**.

**The language pathway contributes almost nothing.** Swapping PubMedBERT for general-domain BERT, for a randomly initialized never-trained BERT, or for random vectors carrying no language at all produces effects that flip sign across training runs — indistinguishable from the 0.0044 Dice noise floor of simply retraining the baseline. Only discarding PubMedBERT's anisotropic embedding *geometry* reliably hurts. Probing the trained model directly, negation, word-order destruction, referent swapping and contentless filler all leave the projected query above 0.94 cosine to the original and the heatmap at ρ = 0.56–1.00; for whole tumor, generic filler *beats* the true clinical description. The query functions as an opaque class identifier that happens to be spelled in English.

**What works is not what we first thought, and it is free.** Size-conditioned prompting (whose reported gain turns out to be a binarizer artifact, vanishing under every calibrated rule), multi-scale ensembling, scale-matched retraining (which a noise probe shows learned resize-interpolation shortcuts, ANOVA p≈4×10⁻²⁵¹) and a uniformity regularizer all fail to beat the plain baseline. A smaller query window does help — but a factorial varying window size and stride independently, which the original sweep never did, reassigns most of that gain to **sampling density rather than receptive field** (+0.076 Dice from sampling more finely at a fixed 32³ window, against +0.020 from shrinking the window at fixed stride). The largest effect in the project is elsewhere: the sliding window gives every voxel it covers the same scalar score, so the heatmap is piecewise-constant over stride³ blocks. Weighting each window's contribution toward its own centre instead — same model, same windows, **no retraining and no extra forward passes** — collapses the tied plateau from 4,096 voxels to 1, is worth **+0.145 Dice** at the original protocol, and takes the previously-at-chance small-ET bin from 0 of 21 patients to 7 of 21.

**The methodological result is the transferable one.** Eight conclusions here were overturned or materially qualified *after* being written up, by controls rather than by significance tests — including this project's own central mechanism, twice. Three components assumed to be neutral instruments turned out to carry results: the binarizer, the coupling of window size to stride, and the rule attributing a window's score to voxels. The lesson: **a shared confound is not a cancelled confound**, and every stage of a pipeline is an instrument until measured otherwise. Every comparison used one binarizer, which we argued made them internally fair; Otsu and every calibrated rule turn out to disagree in *sign* about the same pair of heatmaps, so arms that interact differently with a shared instrument can be ranked backwards by it — invisibly to any amount of paired testing or FDR correction.

**Two documents:**

- [`report_draft.pdf`](report_draft.pdf) — the final report, written as a short research paper (54 pp).
- [`work_log.pdf`](work_log.pdf) — a complete companion account: every one of the seventeen experiments in the order they happened, every script that produced every number, and a plain-language primer that assumes no background in medical imaging, contrastive learning or the statistics used. Also documents the eight conclusions this project had to reverse, and how each was caught.

Between them the two documents carry all 43 figures; [`figures/README.md`](figures/README.md) maps each one to the claim it supports and the script that draws it.

## Repository layout

- `src/` — all code (62 files, 11,471 lines). See [`src/README.md`](src/README.md) for a file-by-file breakdown with line counts.
- `slurm/` — SLURM batch scripts for every training and evaluation job. See [`slurm/README.md`](slurm/README.md).
- `results/` — per-patient CSV outputs from every evaluation run. See [`results/README.md`](results/README.md) for the schema.
- `figures/` — generated report figures. See [`figures/README.md`](figures/README.md).
- `report_draft.md` / `report_draft.pdf` — the final report.
- `work_log.md` / `work_log.pdf` — the complete work log and plain-language companion.

Not included in this repo (see below for how to regenerate):
- `data/` — BraTS2020 raw + preprocessed volumes (~12GB).
- `checkpoints/` — trained model weights (~400MB).
- `logs/` — raw SLURM job stdout (the curated `logs/*_analysis.txt` summaries the report cites *are* tracked).

## Reproducing this project

> **Paths are absolute to the cluster this ran on.** Every script resolves its inputs and outputs
> under `/net/projects/ranalab/rajhansini/nlp_project` — 53 of the 62 files in `src/` contain that
> prefix, as do the `slurm/*.sbatch` job scripts. This was a research pipeline run in one place, not
> a portable package. To run it elsewhere, replace that prefix with your own checkout path
> (`grep -rl '/net/projects/ranalab/rajhansini/nlp_project' src/ slurm/` finds every occurrence);
> nothing else about the code is machine-specific. The steps below otherwise hold as written.

1. **Environment**: Python 3.10, PyTorch 2.4.0+cu121, MONAI 1.4.0, transformers 4.40.0. `requirements.txt` pins every version the reported results were produced under, and documents *why* the torch/MONAI/transformers pins cannot be bumped independently (Section 9 of the report tells the longer story). Install with `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121`.
2. **Data**: download BraTS2020 from Kaggle (`awsaf49/brats20-dataset-training-validation`) into `data/BraTS2020_TrainingData/`.
3. **Preprocess**: `python src/preprocess.py` — normalizes volumes, resizes to 128³, computes per-patient lesion volumes for size-bin stratification.
4. **Text embeddings**: `python src/text_encoder.py` — embeds all region descriptions (base, size-conditioned, and naturalistic variants) with PubMedBERT.
5. **Train the alignment baseline**: `python src/train_baseline.py` (see `slurm/train_baseline.sbatch`). Pass `--seed 1` / `--seed 2` for the cross-seed replications.
6. **Run the P′ check**: `python src/train_pprime_supervised.py` then `python src/evaluate_pprime_supervised.py` — validates the shared pipeline against published BraTS Dice before any result downstream is trusted.
7. **Run RQ1 through RQ12**: see the corresponding `train_rqN.py` / `evaluate_rqN.py` scripts and their matching `slurm/*.sbatch` files. Smoke-test each on the `dev` partition first (`slurm/smoke_test_*.sbatch`).
8. **Statistical analysis**: `python src/analyze_full_family.py` reproduces all 171 paired significance tests with BH-FDR correction under both pooled and per-RQ families. `analyze_seed_replication.py` and `analyze_rq7_multiseed.py` do the cross-run replication checks; `analyze_rq11.py` and `analyze_rq12.py` the threshold and grounding decompositions; `analyze_appendix.py` the eight blocks in the report's Appendix A (IoU, effect sizes, both BH families, checkpoint sensitivity, pointing-rule bounds, Otsu's selected fraction, per-region window optima, and inference cost).
9. **Figures**: `python src/make_figures.py`, then `make_figure4_scale_comparison.py`, `make_overlay_figure.py`, `make_figure5_window_curve.py`, `make_figure_architecture.py`, `make_figure_leaderboard.py`, `make_figure_roadmap.py`, `make_figure_significance_heatmap.py`, `make_figures_dataset_and_evidence.py`, `make_figures_rq7_rq8_rq12.py`, and `make_figures_supplementary.py` / `make_figures_appendix.py` / `make_figures_rq14.py` last (they read the RQ11/RQ12/RQ13/RQ14 CSVs, the training logs and the preprocessed masks). All 43 figures regenerate from `results/`, `logs/` and `data/preprocessed/`; none is drawn by hand.
10. **Build the report**: `pandoc report_draft.md -o report_draft.pdf --pdf-engine=xelatex -V geometry:margin=1in -V fontsize=10pt -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" --toc` (DejaVu is needed for the Unicode maths and Greek in the text).

## Attribution

All code in this repository was written for this project. Third-party dependencies (PyTorch, MONAI, Hugging Face Transformers, PubMedBERT weights) are used via their public APIs/package installs, not vendored.
