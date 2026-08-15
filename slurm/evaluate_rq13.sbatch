#!/bin/bash
# RQ13: re-score one retrained ablation under all five binarizers. $1 = rq2|rq4|rq6,
# $2 = accumulation weighting (optional, "uniform" by default; "gaussian" runs the RQ15 read-out).
#
# Self-resuming. The centre-weighted read-out costs about 3.8x the uniform one per window (measured:
# 59 min against 3h43 at 16^3/stride-4), which puts the scale-matched arms over this partition's
# cap. evaluate_ablation_thresholds.py therefore appends each (patient, region) as it is produced
# and stops cleanly at --max_hours, exiting 42 to mean "work remains"; this script then resubmits
# itself, and the next run skips whatever is already on disk. Any other non-zero exit is a real
# failure and is not retried, so a crash cannot turn into a resubmission loop.
#SBATCH --job-name=nlp_rq13
#SBATCH --partition=general
#SBATCH --account=threedle_group
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/net/projects/ranalab/rajhansini/nlp_project/logs/rq13_%j.out

# Note: no -e here. A non-zero exit from the python step is a signal this script has to inspect
# rather than a reason to abort, and -e would kill the job before the resubmission branch runs.
set -uo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects/ranalab/rajhansini/conda_envs/nlp_project
cd /net/projects/ranalab/rajhansini/nlp_project/src

ARM="$1"
WEIGHTING="${2:-uniform}"
SCRIPT=/net/projects/ranalab/rajhansini/nlp_project/slurm/evaluate_rq13.sbatch

# Stop 20 minutes short of the wall so the pair in flight can finish and the CSV can be flushed.
python evaluate_ablation_thresholds.py --rq "$ARM" --seed 0 \
       --weighting "$WEIGHTING" --max_hours 3.67
STATUS=$?

if [ "$STATUS" -eq 42 ]; then
    echo "=== incomplete at the wall; resubmitting $ARM ($WEIGHTING) ==="
    sbatch "$SCRIPT" "$ARM" "$WEIGHTING"
elif [ "$STATUS" -ne 0 ]; then
    echo "=== failed with status $STATUS; not resubmitting ==="
    exit "$STATUS"
else
    echo "=== $ARM ($WEIGHTING) complete ==="
fi
