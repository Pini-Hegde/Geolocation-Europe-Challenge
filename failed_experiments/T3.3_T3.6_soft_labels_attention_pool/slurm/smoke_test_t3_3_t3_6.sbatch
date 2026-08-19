#!/bin/bash
# GPU smoke test for T3.3 (haversine-smoothed soft cell labels) + T3.6 (query attention
# pooling), combined -- run before committing to the real fold-0 screen. Just runs the real
# training script (src/train_hierarchical.py) for 1 short epoch on a small batch, on the
# current best backbone (EfficientNet-B0), with both new features at their new defaults
# (--pooling attention, --soft-cell-labels) -- catches shape/dtype/AMP issues in ~1min
# instead of burning a multi-hour screen on something broken.
#
# Submit with: sbatch slurm/smoke_test_t3_3_t3_6.sbatch
# Watch with:  squeue -u $USER   /   tail -f slurm_logs/geo_smoke_t33t36_<jobid>.out

#SBATCH --job-name=geo_smoke_t33t36
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.12-base
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
nvidia-smi -L

python -u -m src.train_hierarchical \
  --backbone efficientnet_b0.ra_in1k \
  --folds 0 \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 4 \
  --exp-name T3.3_T3.6_smoke_test

echo "SMOKE TEST PASSED"
echo "Job $SLURM_JOB_ID finished at $(date)"
