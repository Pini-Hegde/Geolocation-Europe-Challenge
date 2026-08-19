"""
Training loop for the hierarchical country -> within-country-cell classifier + offset
regression (`src/models/hierarchical_geo_classifier.py`). Parallel to `src/train.py`
(T2.1's direct regression), kept separate because the target/loss/eval-reconstruction all
genuinely differ, not just a CLI flag's worth.

Target (per fold, from `src/protocol/geo_cells.py`'s country-respecting partition): each
image is labelled with its true country index, a geo-cell id (exactly one country's cells
are reachable, by construction), and its local offset from that cell's centroid in km.

Loss = country cross-entropy + cell cross-entropy (on logits hard-masked to the TRUE
country during training -- teacher forcing, see the model's docstring) + a Huber loss on
the offset, scaled by `OFFSET_SCALE_KM` so all three terms sit on a comparable scale.

Default input resolution is 224px (`--input-size 224`, changed 2026-08-19 from an earlier
384px screen -- that fold-0 run got 225.3km, see Experiment_Design.md). 224px is
EfficientNet-B0's actual native/pretrained resolution (verified via
`timm.data.resolve_data_config`, not assumed) and close to MobileViT-XS's 256px cvnets
recipe; it is NOT FastViT's native resolution (that's 256px, also verified) -- 224px was
still chosen uniformly across all three backbones tested here so the backbone comparison
itself is controlled (same resolution for everyone), at the honest cost of a mismatch for
FastViT specifically. Flag this in the writeup if FastViT underperforms -- it may be this
resolution mismatch, not the architecture, and would need a same-resolution-as-native rerun
to know for sure.

Supports backbone comparison via multiple runs of this script (MobileViT-XS, EfficientNet-B0,
`efficientnet_b0.ra_in1k`; FastViT, `fastvit_t8.apple_dist_in1k` -- the smallest FastViT
variant that fits the param budget, `fastvit_t12` alone is already 6.5M, over budget before
any heads). All three verified compliant at 224px via `check_compliance_hierarchical_model`.

Usage:
    python -m src.train_hierarchical --folds 0 --epochs 15
    python -m src.train_hierarchical --folds 0 1 2 3 4 --epochs 100
    python -m src.train_hierarchical --backbone efficientnet_b0.ra_in1k --folds 2 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import INPUT_SIZE, load_fold_cells, make_dataloaders
from src.geo.spherical import haversine_km_np, local_offset_km_to_latlon_np
from src.models.hierarchical_geo_classifier import HierarchicalGeoClassifierRegressor
from src.protocol.param_check import check_compliance_hierarchical_model

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "outputs" / "experiments"

# Hierarchical-model default resolution -- see module docstring. Deliberately NOT the same
# name as dataset.py's INPUT_SIZE (256, T2.1's default) so it's obvious this is a per-script
# choice, not a global change to the shared dataset pipeline.
DEFAULT_INPUT_SIZE = 224

OFFSET_SCALE_KM = 100.0
HUBER_DELTA = 1.0  # in OFFSET_SCALE_KM units, i.e. a ~100km robustness threshold


def build_optimizer(model: HierarchicalGeoClassifierRegressor, backbone_lr: float, head_lr: float, weight_decay: float):
    head_params = (
        list(model.country_head.parameters())
        + list(model.cls_head.parameters())
        + list(model.offset_head.parameters())
    )
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def compute_losses(cell_logits, offset_pred_scaled, country_logits, cell_id, offset_km_true, country_idx):
    cls_loss = F.cross_entropy(cell_logits, cell_id)
    offset_true_scaled = offset_km_true / OFFSET_SCALE_KM
    offset_loss = F.huber_loss(offset_pred_scaled, offset_true_scaled, delta=HUBER_DELTA)
    country_loss = F.cross_entropy(country_logits, country_idx)
    return cls_loss, offset_loss, country_loss


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, centroids: np.ndarray, device) -> dict:
    model.eval()
    all_lat, all_lng, all_pred_lat, all_pred_lng = [], [], [], []
    n_cell_correct, n_country_correct, n_total = 0, 0, 0
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        # No true_country_idx passed -> model masks by its OWN predicted country, exactly
        # as at real inference time (no ground truth available then either).
        cell_logits, offset_pred_scaled, country_logits = model(images)

        pred_cell = cell_logits.argmax(dim=-1).cpu().numpy()
        pred_country = country_logits.argmax(dim=-1).cpu().numpy()
        pred_offset_km = offset_pred_scaled.cpu().numpy() * OFFSET_SCALE_KM
        centroid = centroids[pred_cell]
        pred_lat, pred_lng = local_offset_km_to_latlon_np(
            pred_offset_km, centroid[:, 0], centroid[:, 1]
        )

        all_lat.append(batch["lat"].numpy())
        all_lng.append(batch["lng"].numpy())
        all_pred_lat.append(pred_lat)
        all_pred_lng.append(pred_lng)

        true_cell = batch["cell_id"].numpy()
        true_country = batch["country_idx"].numpy()
        n_cell_correct += int((pred_cell == true_cell).sum())
        n_country_correct += int((pred_country == true_country).sum())
        n_total += len(true_cell)

    lat = np.concatenate(all_lat)
    lng = np.concatenate(all_lng)
    pred_lat = np.concatenate(all_pred_lat)
    pred_lng = np.concatenate(all_pred_lng)
    err_km = haversine_km_np(lat, lng, pred_lat, pred_lng)
    return {
        "median_km": float(np.median(err_km)),
        "mean_km": float(np.mean(err_km)),
        "frac_lt_200km": float(np.mean(err_km < 200)),
        "frac_lt_750km": float(np.mean(err_km < 750)),
        "cell_top1_acc": n_cell_correct / max(n_total, 1),
        "country_top1_acc": n_country_correct / max(n_total, 1),
    }


def train_one_fold(
    backbone_name: str,
    val_fold: int,
    epochs: int,
    batch_size: int,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    dropout: float,
    offset_weight: float,
    country_weight: float,
    input_size: int,
    num_workers: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_info = load_fold_cells(val_fold)
    n_cells = cell_info["n_cells"]
    centroids = cell_info["centroids"]
    train_loader, val_loader = make_dataloaders(
        val_fold, batch_size=batch_size, num_workers=num_workers,
        input_size=input_size, use_cells=True,
    )

    model = HierarchicalGeoClassifierRegressor(
        backbone_name, n_cells=n_cells, cell_to_country=cell_info["cell_to_country"],
        pretrained=True, dropout=dropout,
    ).to(device)
    optimizer = build_optimizer(model, backbone_lr, head_lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    best_median_km = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_cls_loss = 0.0
        running_offset_loss = 0.0
        running_country_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            cell_id = batch["cell_id"].to(device, non_blocking=True)
            offset_km_true = batch["offset_km"].to(device, non_blocking=True)
            country_idx = batch["country_idx"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                # Teacher forcing: mask by the TRUE country during training (see model
                # docstring) so the cell head's supervision stays clean and unambiguous.
                cell_logits, offset_pred_scaled, country_logits = model(
                    images, true_country_idx=country_idx
                )
                cls_loss, offset_loss, country_loss = compute_losses(
                    cell_logits, offset_pred_scaled, country_logits,
                    cell_id, offset_km_true, country_idx,
                )
                loss = cls_loss + offset_weight * offset_loss + country_weight * country_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_cls_loss += cls_loss.item()
            running_offset_loss += offset_loss.item()
            running_country_loss += country_loss.item()
            n_batches += 1
        scheduler.step()

        val_stats = evaluate(model, val_loader, centroids, device)
        epoch_time = time.time() - t0
        train_cls_loss = running_cls_loss / max(n_batches, 1)
        train_offset_loss = running_offset_loss / max(n_batches, 1)
        train_country_loss = running_country_loss / max(n_batches, 1)
        history.append(
            {
                "epoch": epoch,
                "train_cls_loss": train_cls_loss,
                "train_offset_loss": train_offset_loss,
                "train_country_loss": train_country_loss,
                **val_stats,
                "epoch_time_s": epoch_time,
            }
        )
        print(
            f"[fold {val_fold}] epoch {epoch+1}/{epochs}  cls_loss={train_cls_loss:.4f}  "
            f"offset_loss={train_offset_loss:.4f}  country_loss={train_country_loss:.4f}  "
            f"country_acc={100*val_stats['country_top1_acc']:.1f}%  "
            f"cell_acc={100*val_stats['cell_top1_acc']:.1f}%  "
            f"val_median_km={val_stats['median_km']:.1f}  val_mean_km={val_stats['mean_km']:.1f}  "
            f"<200km={100*val_stats['frac_lt_200km']:.1f}%  ({epoch_time:.1f}s)"
        )

        if val_stats["median_km"] < best_median_km:
            best_median_km = val_stats["median_km"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    final_stats = {
        "best_epoch": best_epoch,
        "best_val_median_km": best_median_km,
        "n_cells": n_cells,
        **history[-1],
    }
    (out_dir / "fold_result.json").write_text(json.dumps(final_stats, indent=2))
    return final_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mobilevit_xs.cvnets_in1k")
    parser.add_argument("--folds", type=int, nargs="+", default=[0], help="val fold id(s) to run")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--backbone-lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--offset-weight", type=float, default=1.0)
    parser.add_argument("--country-weight", type=float, default=1.0)
    parser.add_argument(
        "--input-size", type=int, default=DEFAULT_INPUT_SIZE,
        help=f"default {DEFAULT_INPUT_SIZE}px, up from T2.1's {INPUT_SIZE}px -- see module docstring.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--exp-name", default=None, help="defaults to the backbone name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fold_n_cells = {f: load_fold_cells(f)["n_cells"] for f in args.folds}
    max_n_cells = max(fold_n_cells.values())
    print(f"Per-fold n_cells: {fold_n_cells}, input_size={args.input_size}")
    compliance = check_compliance_hierarchical_model(
        args.backbone, n_cells=max_n_cells, input_size=args.input_size
    )
    print(
        f"Param compliance OK (worst-case n_cells={max_n_cells}): "
        f"{compliance['n_params_total']:,} params (limit {compliance['param_limit']:,})"
    )

    exp_name = args.exp_name or args.backbone.replace("/", "_")
    exp_dir = OUT_DIR_BASE / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "param_compliance.json").write_text(json.dumps(compliance, indent=2))
    config = vars(args) | {
        "fold_n_cells": fold_n_cells,
        "offset_scale_km": OFFSET_SCALE_KM,
        "huber_delta": HUBER_DELTA,
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))

    fold_results = {}
    for val_fold in args.folds:
        fold_dir = exp_dir / f"fold{val_fold}"
        result = train_one_fold(
            backbone_name=args.backbone,
            val_fold=val_fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            offset_weight=args.offset_weight,
            country_weight=args.country_weight,
            input_size=args.input_size,
            num_workers=args.num_workers,
            device=device,
            out_dir=fold_dir,
        )
        fold_results[val_fold] = result

    medians = np.array([r["best_val_median_km"] for r in fold_results.values()])
    means = np.array([r["mean_km"] for r in fold_results.values()])
    cell_accs = np.array([r["cell_top1_acc"] for r in fold_results.values()])
    country_accs = np.array([r["country_top1_acc"] for r in fold_results.values()])
    summary = {
        "backbone": args.backbone,
        "folds_run": args.folds,
        "input_size": args.input_size,
        "cv_median_km_mean": float(medians.mean()),
        "cv_median_km_std": float(medians.std(ddof=1)) if len(medians) > 1 else 0.0,
        "cv_mean_km_mean": float(means.mean()),
        "cv_cell_top1_acc_mean": float(cell_accs.mean()),
        "cv_country_top1_acc_mean": float(country_accs.mean()),
        "per_fold": {k: v for k, v in fold_results.items()},
    }
    # Only write cv_summary.json when THIS run actually covered more than one fold. When
    # folds are trained as separate parallel jobs (one fold per job, same --exp-name --
    # src/aggregate_fold_results.py's use case), each job seeing only its own single fold
    # would otherwise overwrite cv_summary.json with a misleading "5-fold" file that's
    # really just whichever job happened to finish last. Each job's own single-fold result
    # still lives in fold_result.json regardless -- nothing is lost, just not prematurely
    # (and wrongly) aggregated here.
    if len(args.folds) > 1:
        (exp_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(
        f"CV summary ({len(args.folds)} fold(s)): median km = "
        f"{summary['cv_median_km_mean']:.1f} +/- {summary['cv_median_km_std']:.1f}, "
        f"mean km = {summary['cv_mean_km_mean']:.1f}, "
        f"cell acc = {100*summary['cv_cell_top1_acc_mean']:.1f}%, "
        f"country acc = {100*summary['cv_country_top1_acc_mean']:.1f}%"
    )
    if len(args.folds) == 1:
        print(
            "(single-fold run -- cv_summary.json NOT written; run "
            "src/aggregate_fold_results.py once all folds for this exp_name are done)"
        )
    print(f"Artifacts written to {exp_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
