#!/bin/bash
# T3.3 (haversine-smoothed soft cell labels) + T3.6 (query attention pooling), fold-0 quick
# screen -- both changes on top of the current best model (EfficientNet-B0 hierarchical
# country->cell classifier + offset regression, 224px, T3.1: 193.0+/-4.2km full 5-fold).
# Run together (orthogonal changes -- one changes the loss, one changes the feature) rather
# than sequentially, per literature.md's recommendation. Two-stage screening protocol: only
# worth a full 5-fold run if this clears T3.1's 193.0km (single-fold best was better than
# that, so compare against T3.1's own fold-0 number, not the CV mean, once results are in --
# see src/aggregate_fold_results.py / the fold-specific artifact from the T3.1 grid run).
#
# Submit with: sbatch slurm/train_hierarchical_t3_3_t3_6_screen.sbatch
# Watch with:  squeue -u $USER   /   tail -f slurm_logs/geo_t33t36_screen_<jobid>.out
# Cancel with: scancel <jobid>

#SBATCH --job-name=geo_t33t36_screen
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
  --pooling attention \
  --soft-cell-labels \
  --cell-label-tau-km 50.0 \
  --exp-name T3.3_T3.6_efficientnet_b0_224_fold0_screen

echo "Job $SLURM_JOB_ID finished at $(date)"
