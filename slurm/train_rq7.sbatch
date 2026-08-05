#!/bin/bash
# RQ7 text-encoder ablation training. Parametrized by variant, passed as $1.
# Submit with an explicit per-variant job name and log path so runs stay traceable, e.g.:
#   sbatch --job-name=nlp_rq7_train_bertbase \
#          --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq7_train_bertbase_%j.out \
#          slurm/train_rq7.sbatch bertbase
#SBATCH --job-name=nlp_rq7_train
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq7_train_%j.out

set -euo pipefail
VARIANT="$1"
SEED="${2:-0}"   # seed controls the train/val split; 0 reproduces the primary run

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python train_rq7.py --variant "$VARIANT" --seed "$SEED" --epochs 30 --batch_size 32 --num_workers 4
