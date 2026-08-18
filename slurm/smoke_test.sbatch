#!/bin/bash
# Quick GPU sanity check before committing to the full 5-fold job.
# Verifies: allocated a real GPU, .venv's torch/timm see it, the project's own
# modules import cleanly, and the param-compliance gate passes -- all the ways
# the real job could fail in the first 30 seconds instead of after 3 hours.
#
# Submit with: sbatch slurm/smoke_test.sbatch

#SBATCH --job-name=geo_smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load python/3.12-base
source .venv/bin/activate

# GPU compute nodes on Alex appear to have no outbound internet (the first smoke
# test run hung for 5+ min right at timm's pretrained-weight download and got
# TIMEOUT'd). Weights are now pre-cached on the login node (which does have
# internet); force offline mode so this never tries the network.
export HF_HUB_OFFLINE=1

python -u - <<'PY'
import torch, timm
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible inside the job"
print("device:", torch.cuda.get_device_name(0))

from src.protocol.param_check import check_compliance
c = check_compliance("mobilevit_xs.cvnets_in1k")
print("param compliance:", c["n_params_total"], "/", c["param_limit"])
assert c["n_params_total"] <= c["param_limit"]

from src.data.dataset import make_dataloaders
train_loader, val_loader = make_dataloaders(0, batch_size=8, num_workers=2)
batch = next(iter(train_loader))
print("batch image shape:", batch["image"].shape)

from src.models.backbone_regressor import GeoBackboneRegressor
model = GeoBackboneRegressor("mobilevit_xs.cvnets_in1k", pretrained=True, dropout=0.3).cuda()
out = model(batch["image"].cuda())
print("forward pass ok, output shape:", out.shape)

print("SMOKE TEST PASSED")
PY
