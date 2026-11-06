#!/bin/bash
# T3.4 (combinatorial/multi-partition geo-cells) -- fold-0 quick screen. EfficientNet-B0,
# K=2 independent partitions, same 224px/regularization config as T3.1's baseline (only the
# cell-partitioning mechanism is the tested variable). Two-stage screening protocol: only
# worth a full 5-fold run if this clears T3.1's actual fold-0 number (191.7km, cell acc
# 42.6% -- outputs/experiments/T3.1_hier_effnet_224_5fold/fold0/fold_result.json).
#
# Submit with: sbatch slurm/train_multi_partition_fold0_screen.sbatch
# Watch with:  squeue -u $USER   /   tail -f slurm_logs/geo_t34_screen_<jobid>.out

#SBATCH --job-name=geo_t34_screen
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=9
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=praveenhegde1235@gmail.com

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
  --epochs 100 \
  --batch-size 128 \
  --backbone-lr 2e-4 \
  --head-lr 2e-3 \
  --weight-decay 5e-4 \
  --dropout 0.3 \
  --offset-weight 1.0 \
  --country-weight 1.0 \
  --input-size 224 \
  --num-workers 8 \
  --exp-name T3.4_efficientnet_b0_224_fold0_screen

echo "Job $SLURM_JOB_ID finished at $(date)"
