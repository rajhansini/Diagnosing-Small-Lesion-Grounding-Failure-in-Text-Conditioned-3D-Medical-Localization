#!/bin/bash
#SBATCH --job-name=nlp_seed_smoke
#SBATCH --partition=dev
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/seed_smoke_%j.out

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python train_baseline.py --seed 1 --limit_patients 15 --epochs 2 --batch_size 8 --num_workers 2
