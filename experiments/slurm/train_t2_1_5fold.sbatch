#!/bin/bash
# Full 5-fold CV for T2.1 (MobileViT-XS, regularized config) -- Progress Tracker's
# next step. Runs as a real SLURM batch job on a GPU node, NOT on the login node
# and NOT as a backgrounded/nohup'd process: once `sbatch` hands this to slurmctld
# it is scheduled and executed by slurmd on a compute node, fully independent of
# this shell / SSH session. Disconnecting, closing the laptop, etc. does not touch
# it -- only `scancel` or the job finishing does. This is the fix for the earlier
# "outputs/experiments/T2.1_mobilevit_xs_5fold_attempt1_died_ep30" attempt, which
# was run directly on the alex2 login node (no GPU there at all) and got killed.
#
# Submit with: sbatch slurm/train_t2_1_5fold.sbatch
# Watch with:  squeue -u $USER   /   tail -f slurm_logs/geo_t2.1_5fold_<jobid>.out
# Cancel with: scancel <jobid>

#SBATCH --job-name=geo_t2.1_5fold
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=praveenhegde1235@gmail.com

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load python/3.12-base
source .venv/bin/activate

# See slurm/smoke_test.sbatch: GPU compute nodes have no outbound internet, so
# force offline HF Hub mode -- weights must already be cached (they are, via
# the login-node pre-download done before the smoke test).
export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
nvidia-smi -L

# Same regularized config that got fold-0 to 441.8km median (train/val gap 144.1km)
# in outputs/experiments/T2.1_mobilevit_xs_fold0_screen_regularized/ -- now run
# across all 5 folds for the real reportable T2.1 CV number.
python -u -m src.train \
  --backbone mobilevit_xs.cvnets_in1k \
  --folds 0 1 2 3 4 \
  --epochs 100 \
  --batch-size 128 \
  --backbone-lr 2e-4 \
  --head-lr 2e-3 \
  --weight-decay 5e-4 \
  --dropout 0.3 \
  --num-workers 8 \
  --exp-name T2.1_mobilevit_xs_5fold_full

echo "Job $SLURM_JOB_ID finished at $(date)"
