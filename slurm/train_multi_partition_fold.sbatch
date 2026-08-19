#!/bin/bash
# T3.4 (combinatorial/multi-partition geo-cells) -- one fold per job, parameterized, so all 5
# folds run in parallel instead of sequentially. Mirrors train_hierarchical_fold.sbatch's
# pattern (T3.1's backbone-comparison grid).
#
# Required env vars (pass via `sbatch --export=ALL,FOLD=...,EXP_NAME=...`):
#   FOLD      -- val fold id, 0-4
#   EXP_NAME  -- experiment dir name (same across all 5 folds, so
#                src/aggregate_fold_results.py can find fold0/..fold4/ under it afterward)
# Optional: BACKBONE (default efficientnet_b0.ra_in1k), N_PARTITIONS (default 2).
#
# Submit e.g.:
#   sbatch --job-name=geo_t34_f0 --export=ALL,FOLD=0,EXP_NAME=T3.4_efficientnet_b0_224_5fold slurm/train_multi_partition_fold.sbatch
#
# Aggregate afterward (once all 5 folds for one exp_name are COMPLETED):
#   python -m src.aggregate_fold_results --exp-name T3.4_efficientnet_b0_224_5fold

#SBATCH --job-name=geo_t34_fold
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=9
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=praveenhegde1235@gmail.com

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${FOLD:?FOLD env var required}"
: "${EXP_NAME:?EXP_NAME env var required}"
BACKBONE="${BACKBONE:-efficientnet_b0.ra_in1k}"
N_PARTITIONS="${N_PARTITIONS:-2}"

module load python/3.12-base
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "BACKBONE=$BACKBONE FOLD=$FOLD N_PARTITIONS=$N_PARTITIONS EXP_NAME=$EXP_NAME"
nvidia-smi -L

python -u -m src.train_multi_partition \
  --backbone "$BACKBONE" \
  --folds "$FOLD" \
  --n-partitions "$N_PARTITIONS" \
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
  --exp-name "$EXP_NAME"

echo "Job $SLURM_JOB_ID finished at $(date)"
