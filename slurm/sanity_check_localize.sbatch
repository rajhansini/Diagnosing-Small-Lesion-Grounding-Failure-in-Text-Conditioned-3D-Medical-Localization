#!/bin/bash
#SBATCH --job-name=nlp_sanity_localize
#SBATCH --partition=dev
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/sanity_localize_%j.out

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python sanity_check_localize.py
