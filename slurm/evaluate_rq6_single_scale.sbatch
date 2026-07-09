#!/bin/bash
#SBATCH --job-name=nlp_rq6_single
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq6_single_%j.out

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python evaluate_rq6_single_scale.py
