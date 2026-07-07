#!/bin/bash
#SBATCH --job-name=nlp_window12
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/window12_%j.out

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python evaluate_window_sweep.py --window_size 12 --stride 6
