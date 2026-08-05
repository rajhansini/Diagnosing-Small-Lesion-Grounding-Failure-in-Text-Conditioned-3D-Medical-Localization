#!/bin/bash
# Size-stratified evaluation of the P' supervised reference (see src/evaluate_pprime_supervised.py).
#SBATCH --job-name=nlp_pprime_eval
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/pprime_eval_%j.out

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python evaluate_pprime_supervised.py --seed 0
