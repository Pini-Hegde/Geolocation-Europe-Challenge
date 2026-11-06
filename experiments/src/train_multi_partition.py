"""
Training loop for the multi-partition hierarchical country -> within-country-cell classifier
+ offset regression (T3.4, `src/models/multi_partition_geo_classifier.py`). Parallel to
`src/train_hierarchical.py` (T3.1), kept as a separate script rather than a flag on that one
-- T3.1 is the working baseline every candidate is measured against and stays untouched (see
`failed_experiments/T3.3_T3.6_soft_labels_attention_pool/` for why: leaving mode-switch flags
on the baseline file was the pattern used there, reverted after the experiment failed).

Target (per fold, from `src/protocol/geo_cells_multi.py`'s K independent per-country
partitions): each image gets K geo-cell ids (one per partition) and a local offset in km
relative to the BLENDED (spherical-mean) centroid across all K partitions' true cells --
built in `src/data/dataset.py`'s `GeoDataset.__getitem__` (`multi_cell_info` path) and
`blended_centroid_np`.

Loss = country cross-entropy + mean over K of (cell cross-entropy on that partition's logits,
hard-masked to the TRUE country, teacher forcing -- same masking convention as T3.1) + a
Huber loss on the offset (relative to the blended centroid), scaled by `OFFSET_SCALE_KM`.

At eval/inference, each partition's cell is picked independently (argmax, masked by the
model's own predicted country), the K predicted centroids are combined via the same
spherical-mean blend, and the predicted offset is added to that blended point -- this is the
mechanism being tested: two (or more) independent coarse decisions triangulating a finer
effective location than either alone (literature.md §1, CPlaNet), without needing enough data
to populate a directly-finer joint cell space.

Usage:
    python -m src.train_multi_partition --folds 0 --epochs 15
    python -m src.train_multi_partition --folds 0 1 2 3 4 --epochs 100
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

from src.data.dataset import (
    INPUT_SIZE,
    blended_centroid_np,
    load_fold_cells_multi,
    make_dataloaders_multi,
)
from src.geo.spherical import (
    haversine_km_np,
    latlon_to_unit_vector_np,
    local_offset_km_to_latlon_np,
    unit_vector_to_latlon_np,
)
from src.models.multi_partition_geo_classifier import MultiPartitionGeoClassifierRegressor
from src.protocol.param_check import check_compliance_multi_partition_model

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "outputs" / "experiments"

# Same default as train_hierarchical.py -- EfficientNet-B0's native/pretrained resolution,
# see that module's docstring for the full reasoning. Kept identical here on purpose: T3.4
# isolates the cell-partitioning mechanism as the tested variable, not resolution.
DEFAULT_INPUT_SIZE = 224

OFFSET_SCALE_KM = 100.0
HUBER_DELTA = 1.0
DEFAULT_N_PARTITIONS = 2  # matches src/protocol/geo_cells_multi.py's N_PARTITIONS


def build_optimizer(
    model: MultiPartitionGeoClassifierRegressor, backbone_lr: float, head_lr: float, weight_decay: float
):
    head_params = (
        list(model.country_head.parameters())
        + list(model.cls_heads.parameters())
        + list(model.offset_head.parameters())
    )
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def compute_losses(
    cell_logits_list, offset_pred_scaled, country_logits, cell_ids, offset_km_true, country_idx
):
    """`cell_ids`: (B, K) long tensor -- column k is the true cell id under partition k."""
    cls_losses = [
        F.cross_entropy(cell_logits_list[k], cell_ids[:, k]) for k in range(len(cell_logits_list))
    ]
    cls_loss = torch.stack(cls_losses).mean()
    offset_true_scaled = offset_km_true / OFFSET_SCALE_KM
    offset_loss = F.huber_loss(offset_pred_scaled, offset_true_scaled, delta=HUBER_DELTA)
    country_loss = F.cross_entropy(country_logits, country_idx)
    return cls_loss, offset_loss, country_loss


def blend_predicted_centroids(pred_cells: list[np.ndarray], centroids_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """`pred_cells[k]`: (B,) predicted cell ids under partition k. `centroids_list[k]`:
    (n_cells_k, 2) that partition's centroids. Returns blended (lat, lng), each (B,) --
    vectorized batch counterpart of `blended_centroid_np`."""
    vecs = []
    for k, pred_cell_k in enumerate(pred_cells):
        centroid_k = centroids_list[k][pred_cell_k]  # (B, 2)
        vecs.append(latlon_to_unit_vector_np(centroid_k[:, 0], centroid_k[:, 1]))  # (B, 3)
    mean_vec = np.mean(vecs, axis=0)  # (B, 3)
    mean_vec = mean_vec / np.linalg.norm(mean_vec, axis=-1, keepdims=True)
    lat, lng = unit_vector_to_latlon_np(mean_vec)
    return lat, lng


@torch.no_grad()
def evaluate(model: nn.Module, val_loader, centroids_list: list[np.ndarray], device) -> dict:
    model.eval()
    n_partitions = len(centroids_list)
    all_lat, all_lng, all_pred_lat, all_pred_lng = [], [], [], []
    n_cell_correct = np.zeros(n_partitions, dtype=np.int64)
    n_country_correct, n_total = 0, 0
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        cell_logits_list, offset_pred_scaled, country_logits = model(images)

        pred_cells = [cl.argmax(dim=-1).cpu().numpy() for cl in cell_logits_list]
        pred_country = country_logits.argmax(dim=-1).cpu().numpy()
        pred_offset_km = offset_pred_scaled.cpu().numpy() * OFFSET_SCALE_KM

        blend_lat, blend_lng = blend_predicted_centroids(pred_cells, centroids_list)
        pred_lat, pred_lng = local_offset_km_to_latlon_np(pred_offset_km, blend_lat, blend_lng)

        all_lat.append(batch["lat"].numpy())
        all_lng.append(batch["lng"].numpy())
        all_pred_lat.append(pred_lat)
        all_pred_lng.append(pred_lng)

        true_cell_ids = batch["cell_ids"].numpy()  # (B, K)
        true_country = batch["country_idx"].numpy()
        for k in range(n_partitions):
            n_cell_correct[k] += int((pred_cells[k] == true_cell_ids[:, k]).sum())
        n_country_correct += int((pred_country == true_country).sum())
        n_total += len(true_country)

    lat = np.concatenate(all_lat)
    lng = np.concatenate(all_lng)
    pred_lat = np.concatenate(all_pred_lat)
    pred_lng = np.concatenate(all_pred_lng)
    err_km = haversine_km_np(lat, lng, pred_lat, pred_lng)
    cell_top1_acc_per_partition = (n_cell_correct / max(n_total, 1)).tolist()
    return {
        "median_km": float(np.median(err_km)),
        "mean_km": float(np.mean(err_km)),
        "frac_lt_200km": float(np.mean(err_km < 200)),
        "frac_lt_750km": float(np.mean(err_km < 750)),
        "cell_top1_acc_per_partition": cell_top1_acc_per_partition,
        "cell_top1_acc": float(np.mean(cell_top1_acc_per_partition)),
        "country_top1_acc": n_country_correct / max(n_total, 1),
    }


def train_one_fold(
    backbone_name: str,
    val_fold: int,
    n_partitions: int,
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
    cell_info = load_fold_cells_multi(val_fold, n_partitions)
    n_cells_per_partition = cell_info["n_cells"]
    centroids_list = cell_info["centroids"]
    train_loader, val_loader = make_dataloaders_multi(
        val_fold, n_partitions, batch_size=batch_size, num_workers=num_workers,
        input_size=input_size,
    )

    model = MultiPartitionGeoClassifierRegressor(
        backbone_name, n_cells_per_partition=n_cells_per_partition,
        cell_to_country_per_partition=cell_info["cell_to_country"],
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
            cell_ids = batch["cell_ids"].to(device, non_blocking=True)
            offset_km_true = batch["offset_km"].to(device, non_blocking=True)
            country_idx = batch["country_idx"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                cell_logits_list, offset_pred_scaled, country_logits = model(
                    images, true_country_idx=country_idx
                )
                cls_loss, offset_loss, country_loss = compute_losses(
                    cell_logits_list, offset_pred_scaled, country_logits,
                    cell_ids, offset_km_true, country_idx,
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

        val_stats = evaluate(model, val_loader, centroids_list, device)
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
        per_part = ",".join(f"{100*a:.1f}%" for a in val_stats["cell_top1_acc_per_partition"])
        print(
            f"[fold {val_fold}] epoch {epoch+1}/{epochs}  cls_loss={train_cls_loss:.4f}  "
            f"offset_loss={train_offset_loss:.4f}  country_loss={train_country_loss:.4f}  "
            f"country_acc={100*val_stats['country_top1_acc']:.1f}%  "
            f"cell_acc(avg)={100*val_stats['cell_top1_acc']:.1f}% [{per_part}]  "
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
        "n_cells_per_partition": n_cells_per_partition,
        **history[-1],
    }
    (out_dir / "fold_result.json").write_text(json.dumps(final_stats, indent=2))
    return final_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0.ra_in1k")
    parser.add_argument("--folds", type=int, nargs="+", default=[0], help="val fold id(s) to run")
    parser.add_argument("--n-partitions", type=int, default=DEFAULT_N_PARTITIONS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--backbone-lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--offset-weight", type=float, default=1.0)
    parser.add_argument("--country-weight", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--exp-name", default=None, help="defaults to the backbone name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"n_partitions={args.n_partitions}")

    fold_n_cells = {f: load_fold_cells_multi(f, args.n_partitions)["n_cells"] for f in args.folds}
    # Worst-case per-partition cell count across the requested folds (a genuine worst case for
    # the compliance check, same reasoning as train_hierarchical.py's max_n_cells).
    max_n_cells_per_partition = [
        max(fold_n_cells[f][k] for f in args.folds) for k in range(args.n_partitions)
    ]
    print(f"Per-fold n_cells_per_partition: {fold_n_cells}, input_size={args.input_size}")
    compliance = check_compliance_multi_partition_model(
        args.backbone, n_cells_per_partition=max_n_cells_per_partition, input_size=args.input_size
    )
    print(
        f"Param compliance OK (worst-case n_cells_per_partition={max_n_cells_per_partition}): "
        f"{compliance['n_params_total']:,} params (limit {compliance['param_limit']:,})"
    )

    exp_name = args.exp_name or args.backbone.replace("/", "_")
    exp_dir = OUT_DIR_BASE / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "param_compliance.json").write_text(json.dumps(compliance, indent=2))
    config = vars(args) | {
        "fold_n_cells_per_partition": fold_n_cells,
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
            n_partitions=args.n_partitions,
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
        "n_partitions": args.n_partitions,
        "folds_run": args.folds,
        "input_size": args.input_size,
        "cv_median_km_mean": float(medians.mean()),
        "cv_median_km_std": float(medians.std(ddof=1)) if len(medians) > 1 else 0.0,
        "cv_mean_km_mean": float(means.mean()),
        "cv_cell_top1_acc_mean": float(cell_accs.mean()),
        "cv_country_top1_acc_mean": float(country_accs.mean()),
        "per_fold": {k: v for k, v in fold_results.items()},
    }
    if len(args.folds) > 1:
        (exp_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(
        f"CV summary ({len(args.folds)} fold(s)): median km = "
        f"{summary['cv_median_km_mean']:.1f} +/- {summary['cv_median_km_std']:.1f}, "
        f"mean km = {summary['cv_mean_km_mean']:.1f}, "
        f"cell acc (avg) = {100*summary['cv_cell_top1_acc_mean']:.1f}%, "
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
