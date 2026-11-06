#!/bin/bash
# Hierarchical country->cell classifier + offset regression -- fold-0 quick screen.
# MobileViT-XS backbone (kept constant vs T2.1 on purpose -- isolates formulation+resolution
# as the tested variables, see src/train_hierarchical.py's docstring), country-respecting
# geo-cells (src/protocol/geo_cells.py), hard hierarchical masking (not a soft gate --
# src/models/hierarchical_geo_classifier.py), 384px input (up from T2.1's 256px).
# Two-stage screening protocol: only worth a full 5-fold run if this clears the rolled-back
# attempt's ~245-250km by a real margin.
#
# Submit with: sbatch slurm/train_hierarchical_screen.sbatch
# Watch with:  squeue -u $USER   /   tail -f slurm_logs/geo_hier_screen_<jobid>.out
# Cancel with: scancel <jobid>

#SBATCH --job-name=geo_hier_screen
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
  --backbone mobilevit_xs.cvnets_in1k \
  --folds 0 \
  --epochs 100 \
  --batch-size 96 \
  --backbone-lr 2e-4 \
  --head-lr 2e-3 \
  --weight-decay 5e-4 \
  --dropout 0.3 \
  --offset-weight 1.0 \
  --country-weight 1.0 \
  --input-size 384 \
  --num-workers 8 \
  --exp-name T3.1_hierarchical_mobilevit_xs_384_fold0_screen

echo "Job $SLURM_JOB_ID finished at $(date)"
