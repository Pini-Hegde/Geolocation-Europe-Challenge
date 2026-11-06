#!/bin/bash
# Generic GPU sanity check for the hierarchical model with an arbitrary backbone, at 224px --
# run once per NEW backbone before committing to the full parallel (backbone x fold) grid.
# Required env var: BACKBONE (timm model id).

#SBATCH --job-name=geo_smoke_hier_bb
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${BACKBONE:?BACKBONE env var required, e.g. --export=ALL,BACKBONE=efficientnet_b0.ra_in1k}"
module load python/3.12-base
source .venv/bin/activate
export HF_HUB_OFFLINE=1

python -u - <<PY
import torch
backbone = "$BACKBONE"
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "no GPU visible inside the job"
print("device:", torch.cuda.get_device_name(0))
print("backbone:", backbone)

from src.protocol.param_check import check_compliance_hierarchical_model
c = check_compliance_hierarchical_model(backbone, n_cells=96, input_size=224)
print("param compliance:", c["n_params_total"], "/", c["param_limit"])
assert c["n_params_total"] <= c["param_limit"]

from src.data.dataset import make_dataloaders, load_fold_cells
train_loader, val_loader = make_dataloaders(0, batch_size=8, num_workers=2, input_size=224, use_cells=True)
batch = next(iter(train_loader))
print("batch image shape:", batch["image"].shape)

from src.models.hierarchical_geo_classifier import HierarchicalGeoClassifierRegressor
from src.train_hierarchical import compute_losses, evaluate, build_optimizer

cell_info = load_fold_cells(0)
model = HierarchicalGeoClassifierRegressor(
    backbone, n_cells=cell_info["n_cells"], cell_to_country=cell_info["cell_to_country"],
    pretrained=True, dropout=0.3
).cuda()
opt = build_optimizer(model, 2e-4, 2e-3, 5e-4)

cell_logits, offset, country_logits = model(batch["image"].cuda(), true_country_idx=batch["country_idx"].cuda())
print("forward pass ok, cell_logits:", cell_logits.shape, "offset:", offset.shape, "country:", country_logits.shape)

cls_loss, offset_loss, country_loss = compute_losses(
    cell_logits, offset, country_logits, batch["cell_id"].cuda(), batch["offset_km"].cuda(), batch["country_idx"].cuda()
)
print("cls_loss:", cls_loss.item(), "offset_loss:", offset_loss.item(), "country_loss:", country_loss.item())
loss = cls_loss + offset_loss + country_loss
loss.backward()
opt.step()
print("backward + optimizer step ok")

stats = evaluate(model, val_loader, cell_info["centroids"], torch.device("cuda"))
print("eval smoke stats (untrained, sanity only):", stats)

print("SMOKE TEST PASSED:", backbone)
PY
