#!/bin/bash
# Generic seed-replication evaluation driver for RQ2/RQ4/RQ5/RQ6. $1 = rq name, $2 = seed.
#SBATCH --job-name=nlp_seedrepl_eval
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/seedrepl_eval_%j.out

set -euo pipefail
RQ="$1"
SEED="$2"

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python evaluate_${RQ}.py --seed "$SEED"
