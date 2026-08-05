#!/bin/bash
# RQ11: threshold-confound analysis (Otsu vs fixed-percentile vs oracle-volume binarization),
# plus the threshold-independent pointing-game metrics. Frozen RQ1 baseline, no training.
#SBATCH --job-name=nlp_rq11_threshold
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq11_threshold_%j.out

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python evaluate_rq11_threshold_confound.py
