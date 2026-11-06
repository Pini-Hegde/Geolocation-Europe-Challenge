#!/bin/bash
# Isolation screen: T3.3 (haversine-smoothed soft cell labels) ALONE, plain GAP pooling
# (pooling=gap, i.e. the original T3.1 architecture) -- fold-0 only. Written 2026-08-19 after
# the combined T3.3+T3.6 5-fold run regressed vs. T3.1 (210.4+/-7.8km vs. 193.0+/-4.2km, cell
# acc 36.1% vs 41.8%, country acc flat at 72.8%). Compare best_val_median_km / cell_top1_acc
# here against T3.1's actual fold-0 result
# (outputs/experiments/T3.1_hier_effnet_224_5fold/fold0/fold_result.json: 191.7km, cell 42.6%).

#SBATCH --job-name=geo_t33_only_f0
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
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

python -u -m src.train_hierarchical \
  --backbone efficientnet_b0.ra_in1k \
  --folds 0 \
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
  --pooling gap \
  --soft-cell-labels \
  --cell-label-tau-km 50.0 \
  --exp-name T3.3_only_efficientnet_b0_224_fold0_screen

echo "Job $SLURM_JOB_ID finished at $(date)"
