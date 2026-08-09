# `slurm/` — cluster job scripts

SLURM batch scripts used to run every training and evaluation job on the university cluster. All were written for this project; none is adapted from a template or another codebase.

**Totals: 45 scripts, 757 lines** (`wc -l`, including the `#SBATCH` directive blocks). Each is thin by design — the scientific logic lives in `src/`, and these only pin the partition, resources and arguments a given run needs, so that the exact command behind every number in the report is recoverable from version control.

## Cluster conventions these encode

- **`dev` partition, 10-minute cap** — every `smoke_test_*.sbatch` targets this. The workflow throughout the project was: run a truncated version of a job on `dev` first (`--limit_patients`, 2 epochs), confirm it completes and the numbers are sane, only then submit the real job. This caught several bugs that would otherwise have wasted hours of `general` queue time.
- **`general` partition, 4-hour cap** — all real training and evaluation. Jobs that could approach the cap (`train_pprime.sbatch`) take a `--max_hours` argument and stop cleanly before SLURM kills them, so a checkpoint always survives.
- **`--account=threedle_group`** is required on this cluster; jobs are rejected without it.
- **`--gres=gpu:1`** rather than a specific card where the job fits any GPU, because named types (`rtx2080ti`) are frequently drained and a generic request schedules far sooner.
- Logs land in `logs/<name>_%j.out`, where `%j` is the SLURM job ID. Job IDs referenced in the report can be traced back to their log this way.

## Smoke tests (run these first)

| Script | Covers |
|---|---|
| `smoke_test.sbatch` | Baseline training end to end on a handful of patients. |
| `smoke_test_seed.sbatch`, `smoke_test_seed_replication.sbatch` | The seeded-split machinery for cross-seed replication. |
| `smoke_test_rq2/3/4/5/6.sbatch` | Each ablation's training and evaluation path. |
| `smoke_test_rq7_rq8_rq11.sbatch` | The text-encoder ablations, compositionality probes, and threshold decomposition. |
| `smoke_test_rq12.sbatch` | The grounding sweep, including its window-32 reproduction gate against RQ11. |
| `smoke_test_pprime.sbatch` | P′ supervised training and evaluation. |

## Training

| Script | Produces |
|---|---|
| `train_baseline.sbatch` | The contrastive baseline checkpoint used by RQ1 and every frozen-model evaluation. |
| `train_baseline_seed1/2.sbatch` | The same baseline under two further splits, for cross-seed replication. |
| `train_rq2/4/5/6.sbatch` | The four retrained ablation arms. |
| `train_rq7.sbatch` | One text-encoder ablation condition; takes the condition name as `$1`. |
| `train_seed_replication.sbatch` | Generic driver: retrains any of RQ2/4/5/6 under a given seed (`$1` = arm, `$2` = seed). |
| `train_pprime.sbatch` | The P′ supervised 3D U-Net reference. |

## Evaluation

| Script | Produces |
|---|---|
| `evaluate_rq1.sbatch`, `evaluate_rq1_seed1/2.sbatch` | Core size-stratified scores per seed. |
| `evaluate_rq2/3/3b/4/5/6.sbatch` | Each ablation's localization scores. |
| `evaluate_rq6_single_scale.sbatch` | The RQ6 oracle mechanism-isolation test. |
| `evaluate_window6/8/12.sbatch` | The RQ3c window-size curve points. |
| `evaluate_rq7.sbatch`, `evaluate_rq8.sbatch`, `evaluate_rq11.sbatch` | Text-encoder ablations, compositionality probes, threshold decomposition. |
| `evaluate_rq12.sbatch` | Grounding sweep at a given window (`$1` = window size, `$2` = stride). **Stride must match the protocol the corresponding Section 7 result used** — RQ3c's 8³ point used stride 8, RQ1/RQ11 used 32/stride 16 — or the Dice numbers are not comparable. |
| `evaluate_seed_replication.sbatch` | Generic driver: evaluates any of RQ2/4/5/6 at a given seed. |
| `evaluate_rq13.sbatch` | Re-scores one retrained arm (`$1` = `rq2`/`rq4`/`rq6`) under all five binarizers, so a retrained arm's verdict can be separated from the threshold it was read at. |
| `evaluate_pprime.sbatch` | P′ scored against published BraTS ranges and the project's size terciles. |

## Diagnostics

| Script | Covers |
|---|---|
| `sanity_check_localize.sbatch` | Heatmap scores higher inside the true region than outside. |
| `test_rq4_shortcut.sbatch` | Noise probe confirming RQ4 learned resize artifacts. |
| `test_rq6_hub_bias.sbatch` | Whether the uniformity regularizer actually broke the embedding hub. |
| `generate_example_heatmaps.sbatch` | Caches the heatmaps used by the qualitative overlay figure. |

## Line counts, per file

The guidelines ask each directory README to give the purpose and line count of every file. Purposes are grouped by role above; the per-file counts are here.

| Script | Lines | Script | Lines |
|---|---|---|---|
| `evaluate_pprime.sbatch` | 16 | `smoke_test_pprime.sbatch` | 18 |
| `evaluate_rq1.sbatch` | 14 | `smoke_test_rq12.sbatch` | 22 |
| `evaluate_rq11.sbatch` | 18 | `smoke_test_rq2.sbatch` | 14 |
| `evaluate_rq12.sbatch` | 18 | `smoke_test_rq3.sbatch` | 38 |
| `evaluate_rq13.sbatch` | 16 | `smoke_test_rq4.sbatch` | 14 |
| `evaluate_rq1_seed1.sbatch` | 14 | `smoke_test_rq5.sbatch` | 14 |
| `evaluate_rq1_seed2.sbatch` | 14 | `smoke_test_rq6.sbatch` | 14 |
| `evaluate_rq2.sbatch` | 14 | `smoke_test_rq7_rq8_rq11.sbatch` | 36 |
| `evaluate_rq3.sbatch` | 14 | `smoke_test_seed.sbatch` | 14 |
| `evaluate_rq3b.sbatch` | 14 | `smoke_test_seed_replication.sbatch` | 32 |
| `evaluate_rq4.sbatch` | 14 | `test_rq4_shortcut.sbatch` | 14 |
| `evaluate_rq5.sbatch` | 14 | `test_rq6_hub_bias.sbatch` | 14 |
| `evaluate_rq6.sbatch` | 14 | `train_baseline.sbatch` | 14 |
| `evaluate_rq6_single_scale.sbatch` | 14 | `train_baseline_seed1.sbatch` | 14 |
| `evaluate_rq7.sbatch` | 24 | `train_baseline_seed2.sbatch` | 14 |
| `evaluate_rq8.sbatch` | 19 | `train_pprime.sbatch` | 16 |
| `evaluate_seed_replication.sbatch` | 19 | `train_rq2.sbatch` | 14 |
| `evaluate_window12.sbatch` | 14 | `train_rq4.sbatch` | 14 |
| `evaluate_window6.sbatch` | 14 | `train_rq5.sbatch` | 14 |
| `evaluate_window8.sbatch` | 14 | `train_rq6.sbatch` | 14 |
| `generate_example_heatmaps.sbatch` | 14 | `train_rq7.sbatch` | 23 |
| `sanity_check_localize.sbatch` | 14 | `train_seed_replication.sbatch` | 21 |
| `smoke_test.sbatch` | 15 |  |  |

**Total: 45 scripts, 757 lines.**
