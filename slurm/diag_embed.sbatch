#!/bin/bash
#SBATCH --job-name=diag_embed
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.12-base
source .venv/bin/activate
export HF_HUB_OFFLINE=1
nvidia-smi -L
python -u scripts_diag_embed.py
