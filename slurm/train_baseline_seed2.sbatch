#!/bin/bash
#SBATCH --job-name=nlp_baseline_seed2
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/baseline_train_seed2_%j.out

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python train_baseline.py --seed 2 --epochs 30 --batch_size 32 --num_workers 4
