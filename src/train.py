"""
Training loop for a Tier 2 backbone-regressor candidate (Experiment_Design.md Part 2).

Implements the two-stage screening protocol: pass `--folds 0` (or a couple of fold ids) for
a quick screen, `--folds 0 1 2 3 4` for the full 5-fold run reported in the results log.
Trains with `angular_loss` (spherical-vector target, Experiment_Design.md §3), validates
each epoch with the real evaluation metric (`haversine_km_np`, matching CHALLENGE.md's
median/mean-km scoring), and reports mean+-std of median km across whichever folds were run.

Usage:
    python -m src.train --backbone mobilevit_xs.cvnets_in1k --folds 0 --epochs 15
    python -m src.train --backbone mobilevit_xs.cvnets_in1k --folds 0 1 2 3 4 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import make_dataloaders
from src.geo.spherical import angular_loss, haversine_km_np, unit_vector_to_latlon_np
from src.models.backbone_regressor import GeoBackboneRegressor
from src.protocol.param_check import check_compliance

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "outputs" / "experiments"


def build_optimizer(model: GeoBackboneRegressor, backbone_lr: float, head_lr: float, weight_decay: float):
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, device) -> dict:
    model.eval()
    all_lat, all_lng, all_pred_lat, all_pred_lng = [], [], [], []
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        pred_vec = model(images)
        pred_lat, pred_lng = unit_vector_to_latlon_np(pred_vec.cpu().numpy())
        all_lat.append(batch["lat"].numpy())
        all_lng.append(batch["lng"].numpy())
        all_pred_lat.append(pred_lat)
        all_pred_lng.append(pred_lng)

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
    num_workers: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_dataloaders(val_fold, batch_size=batch_size, num_workers=num_workers)

    model = GeoBackboneRegressor(backbone_name, pretrained=True, dropout=dropout).to(device)
    optimizer = build_optimizer(model, backbone_lr, head_lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    best_median_km = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target_vec"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(images)
                loss = angular_loss(pred, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_stats = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        train_loss = running_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "train_angular_loss": train_loss, **val_stats, "epoch_time_s": epoch_time})
        print(
            f"[fold {val_fold}] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_median_km={val_stats['median_km']:.1f}  val_mean_km={val_stats['mean_km']:.1f}  "
            f"<200km={100*val_stats['frac_lt_200km']:.1f}%  ({epoch_time:.1f}s)"
        )

        if val_stats["median_km"] < best_median_km:
            best_median_km = val_stats["median_km"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    final_stats = {"best_epoch": best_epoch, "best_val_median_km": best_median_km, **history[-1]}
    (out_dir / "fold_result.json").write_text(json.dumps(final_stats, indent=2))
    return final_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mobilevit_xs.cvnets_in1k")
    parser.add_argument("--folds", type=int, nargs="+", default=[0], help="val fold id(s) to run")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--exp-name", default=None, help="defaults to the backbone name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Gate: fail loudly before spending any GPU time if the model is over budget.
    compliance = check_compliance(args.backbone)
    print(
        f"Param compliance OK: {compliance['n_params_total']:,} params "
        f"(limit {compliance['param_limit']:,})"
    )

    exp_name = args.exp_name or args.backbone.replace("/", "_")
    exp_dir = OUT_DIR_BASE / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "param_compliance.json").write_text(json.dumps(compliance, indent=2))
    (exp_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

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
            num_workers=args.num_workers,
            device=device,
            out_dir=fold_dir,
        )
        fold_results[val_fold] = result

    medians = np.array([r["best_val_median_km"] for r in fold_results.values()])
    means = np.array([r["mean_km"] for r in fold_results.values()])
    summary = {
        "backbone": args.backbone,
        "folds_run": args.folds,
        "cv_median_km_mean": float(medians.mean()),
        "cv_median_km_std": float(medians.std(ddof=1)) if len(medians) > 1 else 0.0,
        "cv_mean_km_mean": float(means.mean()),
        "per_fold": {k: v for k, v in fold_results.items()},
    }
    (exp_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(
        f"CV summary ({len(args.folds)} fold(s)): median km = "
        f"{summary['cv_median_km_mean']:.1f} +/- {summary['cv_median_km_std']:.1f}, "
        f"mean km = {summary['cv_mean_km_mean']:.1f}"
    )
    print(f"Artifacts written to {exp_dir.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
