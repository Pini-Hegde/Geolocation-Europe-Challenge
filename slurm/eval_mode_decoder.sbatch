#!/bin/bash
# Zero-parameter T4.1b decoder sweep on the completed T4.1 fold-0 checkpoint.
#SBATCH --job-name=geo_mode_eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=praveenhegde1235@gmail.com

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.12-base
source .venv/bin/activate
export HF_HUB_OFFLINE=1

python -u -m src.eval_mode_decoder \
  --checkpoint outputs/experiments/T4.1_retrieval_ra4_288_fold0_screen/fold0/best_model.pt \
  --output outputs/experiments/T4.1_retrieval_ra4_288_fold0_screen/mode_decoder_results.json \
  --fold 0 --n-partitions 3 --input-size 288 --batch-size 64 --num-workers 8
