#!/bin/bash
# RQ8: compositionality / negation probe on the frozen RQ1 baseline. Five text conditions per
# region, each requiring a full sliding-window pass, plus Spearman heatmap correlations -- so this
# is roughly 5x the cost of a standard RQ1 evaluation.
#SBATCH --job-name=nlp_rq8_compositionality
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq8_compositionality_%j.out

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python evaluate_rq8_compositionality.py
