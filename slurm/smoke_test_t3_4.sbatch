#!/bin/bash
# GPU smoke test for T3.4 (combinatorial/multi-partition geo-cells) -- runs the real
# training script for 1 short epoch on a small batch, EfficientNet-B0, K=2 partitions.
# Run on `a40` (the only GPU generation validated so far in this project -- `rtxpro6k`'s
# Blackwell GPUs caused a silent 10-min hang during the T3.3/T3.6 attempt, see
# failed_experiments/T3.3_T3.6_soft_labels_attention_pool/README.md).
#
# Submit with: sbatch slurm/smoke_test_t3_4.sbatch

#SBATCH --job-name=geo_smoke_t34
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

python -u -m src.train_multi_partition \
  --backbone efficientnet_b0.ra_in1k \
  --folds 0 \
  --n-partitions 2 \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 4 \
  --exp-name T3.4_smoke_test

echo "SMOKE TEST PASSED"
echo "Job $SLURM_JOB_ID finished at $(date)"
