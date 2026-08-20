#!/bin/bash
# Soft hierarchical + geographic retrieval model. One validation fold per job.
# Required: FOLD, EXP_NAME. Optional settings are exposed as environment variables.

#SBATCH --job-name=geo_retrieval
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
BACKBONE="${BACKBONE:-efficientnet_b0.ra4_e3600_r224_in1k}"
EPOCHS="${EPOCHS:-60}"
INPUT_SIZE="${INPUT_SIZE:-288}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GEO_WEIGHT="${GEO_WEIGHT:-0.2}"

module load python/3.12-base
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname): fold=$FOLD exp=$EXP_NAME backbone=$BACKBONE"
nvidia-smi -L

python -u -m src.train_retrieval_geo \
  --backbone "$BACKBONE" \
  --folds "$FOLD" \
  --n-partitions 3 \
  --embedding-dim 256 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --backbone-lr 1e-4 \
  --head-lr 1e-3 \
  --weight-decay 1e-3 \
  --dropout 0.3 \
  --country-weight 1.0 \
  --offset-weight 0.5 \
  --geo-weight "$GEO_WEIGHT" \
  --geo-scale-km 250 \
  --embedding-temperature 0.1 \
  --retrieval-weights 0 2 4 8 \
  --top-ks 1 5 10 \
  --vote-temperature 1.0 \
  --input-size "$INPUT_SIZE" \
  --num-workers 8 \
  --exp-name "$EXP_NAME"

echo "Job $SLURM_JOB_ID finished at $(date)"
