#!/bin/bash
#SBATCH --job-name=diag_cand
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; module load python/3.12-base; source .venv/bin/activate
export HF_HUB_OFFLINE=1; python -u scripts_diag_candidate_oracle.py
