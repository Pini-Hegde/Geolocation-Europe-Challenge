#!/bin/bash
#SBATCH --job-name=geo_erranalysis
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
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi -L
python -u -m src.error_analysis_t4_1c --exp-name T4.1c_geo_supcon_5fold
echo "Job $SLURM_JOB_ID finished at $(date)"
