#!/bin/bash
# T4.1b shared-weight global/local model. Required: FOLD, EXP_NAME.
#SBATCH --job-name=geo_multiview
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=praveenhegde1235@gmail.com

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${FOLD:?FOLD env var required}"
: "${EXP_NAME:?EXP_NAME env var required}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BACKBONE_LR="${BACKBONE_LR:-1e-4}"
HEAD_LR="${HEAD_LR:-1e-3}"
GEO_WEIGHT="${GEO_WEIGHT:-0.2}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.1}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.0}"
SAMPLES_PER_CELL="${SAMPLES_PER_CELL:-4}"
CELLS_DIR="${CELLS_DIR:-}"
N_PARTITIONS="${N_PARTITIONS:-3}"
CLUSTER_WEIGHT="${CLUSTER_WEIGHT:-1.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
INPUT_SIZE="${INPUT_SIZE:-288}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
module load python/3.12-base
source .venv/bin/activate
export HF_HUB_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname): fold=$FOLD exp=$EXP_NAME"
nvidia-smi -L
python -u -m src.train_multiview_retrieval_geo \
  --folds "$FOLD" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --backbone-lr "$BACKBONE_LR" --head-lr "$HEAD_LR" \
  --geo-weight "$GEO_WEIGHT" --consistency-weight "$CONSISTENCY_WEIGHT" \
  --contrastive-weight "$CONTRASTIVE_WEIGHT" --samples-per-cell "$SAMPLES_PER_CELL" \
  --cells-dir "$CELLS_DIR" --n-partitions "$N_PARTITIONS" --cluster-weight "$CLUSTER_WEIGHT" \
  --grad-clip-norm "$GRAD_CLIP_NORM" --init-checkpoint "$INIT_CHECKPOINT" \
  --eval-batch-size 64 --input-size "$INPUT_SIZE" --num-workers 8 \
  --exp-name "$EXP_NAME"
echo "Job $SLURM_JOB_ID finished at $(date)"
