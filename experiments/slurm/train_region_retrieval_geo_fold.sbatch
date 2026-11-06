#!/bin/bash
# T4.1d region-aware retrieval model. Required: FOLD, EXP_NAME, INIT_CHECKPOINT (continues
# from a trained T4.1c checkpoint -- region_head loads fresh via strict=False, see
# src/train_region_retrieval_geo.py).
#SBATCH --job-name=geo_region
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
: "${INIT_CHECKPOINT:?INIT_CHECKPOINT env var required}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BACKBONE_LR="${BACKBONE_LR:-2e-5}"
HEAD_LR="${HEAD_LR:-2e-4}"
REGION_HEAD_LR="${REGION_HEAD_LR:-1e-3}"
REGION_WEIGHT="${REGION_WEIGHT:-1.0}"
REGION_SCORE_WEIGHT="${REGION_SCORE_WEIGHT:-1.0}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.2}"
module load python/3.12-base
source .venv/bin/activate
export HF_HUB_OFFLINE=1
echo "Job $SLURM_JOB_ID on $(hostname): fold=$FOLD exp=$EXP_NAME"
nvidia-smi -L
python -u -m src.train_region_retrieval_geo \
  --folds "$FOLD" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --backbone-lr "$BACKBONE_LR" --head-lr "$HEAD_LR" --region-head-lr "$REGION_HEAD_LR" \
  --region-weight "$REGION_WEIGHT" --region-score-weight "$REGION_SCORE_WEIGHT" \
  --contrastive-weight "$CONTRASTIVE_WEIGHT" --init-checkpoint "$INIT_CHECKPOINT" \
  --eval-batch-size 64 --input-size 288 --num-workers 8 \
  --exp-name "$EXP_NAME"
echo "Job $SLURM_JOB_ID finished at $(date)"
