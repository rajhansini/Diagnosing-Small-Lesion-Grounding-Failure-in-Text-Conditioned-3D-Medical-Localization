#!/bin/bash
# RQ13: re-score one retrained ablation under all five binarizers. $1 = rq2|rq4|rq6
#SBATCH --job-name=nlp_rq13
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq13_%j.out

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python evaluate_ablation_thresholds.py --rq "$1" --seed 0
