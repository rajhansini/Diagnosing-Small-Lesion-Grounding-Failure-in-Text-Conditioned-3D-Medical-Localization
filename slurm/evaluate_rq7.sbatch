#!/bin/bash
# RQ7 text-encoder ablation evaluation. $1 = variant, $2 = checkpoint (last|best, default last).
# 'last' is the pre-registered primary comparison, matching baseline_aligner.pt's own selection rule.
#   sbatch --job-name=nlp_rq7_eval_bertbase \
#          --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq7_eval_bertbase_%j.out \
#          slurm/evaluate_rq7.sbatch bertbase last
#SBATCH --job-name=nlp_rq7_eval
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq7_eval_%j.out

set -euo pipefail
VARIANT="$1"
CKPT="${2:-last}"
SEED="${3:-0}"   # must match the seed the checkpoint was trained with

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python evaluate_rq7.py --variant "$VARIANT" --ckpt "$CKPT" --seed "$SEED"
