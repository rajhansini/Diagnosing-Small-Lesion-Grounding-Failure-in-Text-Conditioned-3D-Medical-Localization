#!/bin/bash
# P' supervised segmentation reference point (see src/train_pprime_supervised.py).
#SBATCH --job-name=nlp_pprime_train
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/pprime_train_%j.out

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python train_pprime_supervised.py --epochs 200 --batch_size 2 --num_workers 4 --max_hours 3.4
