#!/bin/bash
# Generic, parameterized single-fold hierarchical training job -- one fold, one backbone,
# submitted separately per (backbone, fold) pair so all folds for all backbones can run in
# parallel across the cluster instead of sequentially inside one job. Backbone comparison at
# a uniform 224px (src/train_hierarchical.py's new default, see that module's docstring for
# the EfficientNet-B0-native-but-not-FastViT-native caveat).
#
# Required env vars (pass via `sbatch --export=ALL,BACKBONE=...,FOLD=...,EXP_NAME=...`):
#   BACKBONE  -- timm model id, e.g. mobilevit_xs.cvnets_in1k
#   FOLD      -- val fold id, 0-4
#   EXP_NAME  -- experiment dir name (same across all 5 folds of one backbone, so
#                src/aggregate_fold_results.py can find fold0/..fold4/ under it afterward)
#
# Submit all 15 (3 backbones x 5 folds) with the loop in the commit message / session log,
# or by hand:
#   sbatch --job-name=geo_hier_mvit_f0 --export=ALL,BACKBONE=mobilevit_xs.cvnets_in1k,FOLD=0,EXP_NAME=T3.1_hier_mobilevit_xs_224_5fold slurm/train_hierarchical_fold.sbatch
#
# Aggregate afterward (once all 5 folds for one exp_name are COMPLETED):
#   python -m src.aggregate_fold_results --exp-name T3.1_hier_mobilevit_xs_224_5fold

#SBATCH --job-name=geo_hier_fold
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=praveenhegde1235@gmail.com

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${BACKBONE:?BACKBONE env var required, e.g. --export=ALL,BACKBONE=mobilevit_xs.cvnets_in1k,FOLD=0,EXP_NAME=...}"
: "${FOLD:?FOLD env var required}"
: "${EXP_NAME:?EXP_NAME env var required}"

module load python/3.12-base
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "BACKBONE=$BACKBONE FOLD=$FOLD EXP_NAME=$EXP_NAME"
nvidia-smi -L

python -u -m src.train_hierarchical \
  --backbone "$BACKBONE" \
  --folds "$FOLD" \
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
