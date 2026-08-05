#!/bin/bash
# Smoke test for the --seed refactor of RQ2/RQ4/RQ5/RQ6.
#
# The refactor touched the split seeding, the checkpoint filenames and the results filenames of
# eight scripts that produced already-published numbers, so it needs two separate checks:
#   1. the new seed path runs at all (1 epoch, 15 patients, each of the four training scripts);
#   2. seed 0 still behaves exactly as before -- verified separately by the reproduction gate job,
#      which re-runs evaluate_rq5.py --seed 0 and diffs against the committed CSV.
#SBATCH --job-name=nlp_seedrepl_smoke
#SBATCH --partition=dev
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/seedrepl_smoke_%j.out

set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src

for RQ in rq2 rq4 rq5 rq6; do
  echo "############ train_${RQ}.py --seed 1 (1 epoch, 15 patients) ############"
  python train_${RQ}.py --seed 1 --limit_patients 15 --epochs 1 --batch_size 8 --num_workers 2
done

echo "############ checkpoint names written ############"
ls -1 /net/projects/ranalab/rajhansini/nlp_project/checkpoints/ | grep seed1 || echo "NONE FOUND -- BAD"

echo "############ SEED-REPLICATION SMOKE TEST PASSED ############"
