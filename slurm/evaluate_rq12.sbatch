#!/bin/bash
# RQ12 grounding sweep at one window size. $1 = window_size, $2 = stride,
# $3 = accumulation weighting (optional, "uniform" by default).
# Stride must match the protocol the corresponding Section 7 result used, or the Dice numbers are
# not comparable: RQ3c's 8^3 point used stride 8, and RQ1/RQ11 used 32/stride 16.
# Window and stride are independent settings and the RQ14 factorial varies them separately; do not
# assume stride = window/2, which every condition up to RQ13 happened to use.
#SBATCH --job-name=nlp_rq12
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq12_w%j.out

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python evaluate_grounding_sweep.py --window_size "$1" --stride "$2" --weighting "${3:-uniform}"
