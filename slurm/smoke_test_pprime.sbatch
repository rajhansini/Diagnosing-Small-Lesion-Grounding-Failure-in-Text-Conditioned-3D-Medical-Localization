#!/bin/bash
# 10-minute correctness smoke test for the P' pipeline before committing a full 4h GPU run.
#SBATCH --job-name=nlp_pprime_smoke
#SBATCH --partition=dev
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/pprime_smoke_%j.out

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src

echo "### P' training smoke test (4 patients, 2 epochs) ###"
python train_pprime_supervised.py --epochs 2 --eval_every 1 --limit_patients 20 --num_workers 2
