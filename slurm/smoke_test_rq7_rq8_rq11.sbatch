#!/bin/bash
# Combined smoke test for the new RQ7/RQ8/RQ11 code paths, on the 10-minute dev partition.
# Exercises every new script end to end at tiny scale BEFORE committing GPU hours to the real runs:
#   - one training variant for 1 epoch on 15 patients (checks the text-variant load + training loop)
#   - RQ7 evaluation on 2 patients (checks checkpoint load + localization + CSV write)
#   - RQ11 on 2 patients (checks all 5 binarization rules, the distance transform, the summaries)
#   - RQ8 on 2 patients (checks all 5 probe conditions and the Spearman correlations)
# `set -e` means the job fails loudly on the first broken path rather than silently continuing.
#SBATCH --job-name=nlp_rq7_rq8_rq11_smoke
#SBATCH --partition=dev
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq7_rq8_rq11_smoke_%j.out

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src

echo "############ 1/4 RQ7 training (randvec_ortho, 1 epoch, 15 patients) ############"
python train_rq7.py --variant randvec_ortho --limit_patients 15 --epochs 1 --batch_size 8 --num_workers 2

echo "############ 2/4 RQ7 evaluation (randvec_ortho, 2 patients) ############"
python evaluate_rq7.py --variant randvec_ortho --ckpt last --limit_patients 2

echo "############ 3/4 RQ11 threshold confound (2 patients) ############"
python evaluate_rq11_threshold_confound.py --limit_patients 2

echo "############ 4/4 RQ8 compositionality probe (2 patients) ############"
python evaluate_rq8_compositionality.py --limit_patients 2

echo "############ SMOKE TEST PASSED ############"
