#!/bin/bash
# Generic seed-replication training driver for RQ2/RQ4/RQ5/RQ6. $1 = rq name, $2 = seed.
# Uses a script file rather than sbatch --wrap because --wrap executes under /bin/sh (dash),
# where the `source` builtin does not exist and conda activation silently fails.
#SBATCH --job-name=nlp_seedrepl_train
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/seedrepl_train_%j.out

set -euo pipefail
RQ="$1"
SEED="$2"

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src
python train_${RQ}.py --seed "$SEED" --epochs 30 --batch_size 32 --num_workers 4
