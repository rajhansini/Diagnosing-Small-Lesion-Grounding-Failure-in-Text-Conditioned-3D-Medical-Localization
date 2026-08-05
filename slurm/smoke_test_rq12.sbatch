#!/bin/bash
# 10-minute correctness smoke test for the RQ12 grounding sweep.
#SBATCH --job-name=nlp_rq12_smoke
#SBATCH --partition=dev
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq12_smoke_%j.out

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src

echo "### RQ12 window=32 (must match RQ11) on 3 patients ###"
python evaluate_grounding_sweep.py --window_size 32 --stride 16 --limit_patients 3

echo
echo "### RQ12 window=8 on 2 patients ###"
python evaluate_grounding_sweep.py --window_size 8 --stride 4 --limit_patients 2
