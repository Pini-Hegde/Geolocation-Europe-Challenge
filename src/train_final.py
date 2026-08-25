"""Train the single submitted model on ALL labelled data (no held-out validation fold).

Same architecture/losses/optimizer as `train_multiview_retrieval_geo.py` (the winning T4.1c/
T5.1 recipe, chosen from the 5-fold CV screen) -- the only two differences are:

1. The geo-cell partition is built from ALL of `train_labels.csv`, not 4/5 of it (via
   `src.protocol.geo_cells_multi.build_all_partitions` called with a sentinel fold id that no
   real row has, so `build_fold`'s `df["fold"] != fold_id` filter keeps every row as "train").
2. There is no held-out fold to validate against, so this trains for a FIXED epoch count
   (chosen from the CV screen's best-epoch pattern) rather than early-stopping on val median-km.
   It still prints an in-sample (leaky, informational only) train-set decode every
   `--eval-every` epochs so a stalled/diverged run is visible, but that number must never be
   quoted as the model's expected holdout performance -- use the CV numbers for that.

`--init-checkpoint` warm-starts from any prior fold checkpoint: the backbone/country_head/
offset_head/embedding_head weights are shape-identical across every cell partition (only
`cls_heads` depend on the partition's per-partition cell count) and are loaded; `cls_heads`
are always fresh (`strict=False`, shape-mismatched keys skipped) since the full-data partition
has a different cell count than any single fold's partition.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from timm.data import resolve_data_config
from torch.utils.data import DataLoader

from src.data.dataset import GeoDataset, load_fold_cells_multi, load_labelled_df
from src.data.multiview import GeoBalancedBatchSampler, GlobalLocalTransform
from src.geo.retrieval import conditional_cell_log_probs, spherical_topk_decode
from src.geo.spherical import haversine_km_np
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier
from src.protocol import geo_cells as geo_cells_mod
from src.protocol.geo_cells import load_joined
from src.protocol.geo_cells_multi import N_PARTITIONS as MAX_BUILT_PARTITIONS
from src.protocol.geo_cells_multi import build_all_partitions
from src.train_multiview_retrieval_geo import extract_gallery, hierarchy_scores, supervised_geo_contrastive_loss
from src.train_retrieval_geo import PARAM_LIMIT, build_optimizer, geo_distribution_loss, masked_cell_loss

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "outputs" / "experiments"
FULL_CELLS_DIR = REPO_ROOT / "outputs" / "protocol" / "geo_cells_multi_full"
FULL_FOLD_ID = -1  # sentinel: no row's "fold" column is ever -1, so build_fold keeps 100%


def ensure_full_data_cells(n_partitions: int, min_cell_size: int, max_cell_size: int) -> None:
    if n_partitions > MAX_BUILT_PARTITIONS:
        raise ValueError(f"only {MAX_BUILT_PARTITIONS} independent partitions are built")
    marker = FULL_CELLS_DIR / f"fold{FULL_FOLD_ID}_p{n_partitions - 1}_cells.json"
    if marker.exists():
        return
    df = load_joined()
    build_all_partitions(
        df, out_dir=FULL_CELLS_DIR, folds=[FULL_FOLD_ID],
        min_cell_size=min_cell_size, max_cell_size=max_cell_size,
    )


def make_full_loaders(args):
    ensure_full_data_cells(args.n_partitions, args.min_cell_size, args.max_cell_size)
    df = load_labelled_df()  # every row keeps its real 0-4 fold id (unused for splitting here)
    cell_info = load_fold_cells_multi(FULL_FOLD_ID, args.n_partitions, cells_dir=FULL_CELLS_DIR)
    probe = timm.create_model(args.backbone, pretrained=False)
    cfg = resolve_data_config({}, model=probe)
    del probe
    preprocess = {
        "mean": list(cfg["mean"]), "std": list(cfg["std"]),
        "crop_pct": float(cfg.get("crop_pct", 0.9)),
        "views": "global_plus_four_corner_crops", "local_fraction": args.local_fraction,
    }
    train_ds = GeoDataset(df, train=True, input_size=args.input_size, multi_cell_info=cell_info)
    train_ds.transform = GlobalLocalTransform(
        args.input_size, cfg["mean"], cfg["std"], preprocess["crop_pct"], True, args.local_fraction,
    )
    gallery_ds = GeoDataset(df, train=False, input_size=args.input_size, multi_cell_info=cell_info)
    gallery_ds.transform = GlobalLocalTransform(
        args.input_size, cfg["mean"], cfg["std"], preprocess["crop_pct"], False, args.local_fraction,
    )
    sampler = GeoBalancedBatchSampler(
        train_ds.df["filename"].tolist(), cell_info["assignments"][0],
        args.batch_size, args.samples_per_cell,
    )
    worker_args = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **worker_args)
    gallery_loader = DataLoader(
        gallery_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **worker_args
    )
    return train_loader, gallery_loader, cell_info, preprocess


@torch.no_grad()
def in_sample_decode_check(model, gallery_loader, gallery, device, sample_batches: int = 8):
    """Leaky in-sample sanity check ONLY (every gallery point can retrieve itself) -- reports
    whether training is progressing, not expected holdout performance. See module docstring."""
    model.eval()
    errors = []
    for i, batch in enumerate(gallery_loader):
        if i >= sample_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        cell_logits, _, country_logits, embedding, views = model(images)
        hierarchy = hierarchy_scores(model, country_logits, cell_logits, gallery)
        similarity = embedding.float() @ gallery["embeddings"].float().T
        scores = hierarchy + 4.0 * similarity
        pred_lat, pred_lng, _ = spherical_topk_decode(scores, gallery["unit_vectors"], top_k=10)
        true_lat, true_lng = batch["lat"].numpy(), batch["lng"].numpy()
        errors.extend(haversine_km_np(true_lat, true_lng, pred_lat.cpu().numpy(), pred_lng.cpu().numpy()))
    return float(np.median(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0.ra4_e3600_r224_in1k")
    parser.add_argument("--n-partitions", type=int, default=3)
    parser.add_argument("--min-cell-size", type=int, default=geo_cells_mod.MIN_CELL_SIZE)
    parser.add_argument("--max-cell-size", type=int, default=geo_cells_mod.MAX_CELL_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--samples-per-cell", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--country-weight", type=float, default=1.0)
    parser.add_argument("--offset-weight", type=float, default=0.5)
    parser.add_argument("--geo-weight", type=float, default=0.2)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--geo-scale-km", type=float, default=250.0)
    parser.add_argument("--embedding-temperature", type=float, default=0.1)
    parser.add_argument("--local-fraction", type=float, default=0.64)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--exp-name", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT_DIR_BASE / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, gallery_loader, cell_info, preprocess = make_full_loaders(args)
    model = MultiViewRetrievalGeoClassifier(
        args.backbone, cell_info["n_cells"], cell_info["cell_to_country"],
        embedding_dim=args.embedding_dim, pretrained=not bool(args.init_checkpoint),
        dropout=args.dropout,
    ).to(device)
    if args.init_checkpoint:
        src_state = torch.load(args.init_checkpoint, map_location=device)
        dst_state = model.state_dict()
        compatible = {
            k: v for k, v in src_state.items() if k in dst_state and dst_state[k].shape == v.shape
        }
        skipped = sorted(set(src_state) - set(compatible))
        model.load_state_dict(compatible, strict=False)
        print(
            f"Warm-started {len(compatible)}/{len(dst_state)} tensors from {args.init_checkpoint} "
            f"(skipped, shape-mismatched: {skipped})", flush=True,
        )
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > PARAM_LIMIT:
        raise RuntimeError(f"model has {n_params:,} parameters; limit is {PARAM_LIMIT:,}")
    print(f"n_params_total={n_params:,} (limit {PARAM_LIMIT:,})", flush=True)

    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    for epoch in range(args.epochs):
        model.train()
        started = time.time()
        totals, n_batches = np.zeros(6), 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            cells_true = batch["cell_ids"].to(device, non_blocking=True)
            countries = batch["country_idx"].to(device, non_blocking=True)
            offsets = batch["offset_km"].to(device, non_blocking=True)
            lat, lng = batch["lat"].to(device), batch["lng"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, offset, country_logits, embedding, views = model(images)
                cell_loss = masked_cell_loss(model, logits, cells_true, countries)
                country_loss = F.cross_entropy(country_logits, countries)
                offset_loss = F.huber_loss(offset, offsets / 100.0, delta=1.0)
                geo_loss = geo_distribution_loss(
                    embedding, lat, lng, args.geo_scale_km, args.embedding_temperature
                )
                consistency_loss = (1 - (views * embedding[:, None]).sum(dim=-1)).mean()
                contrastive_loss = supervised_geo_contrastive_loss(
                    embedding, cells_true, args.contrastive_temperature
                )
                loss = (cell_loss + args.country_weight * country_loss
                        + args.offset_weight * offset_loss + args.geo_weight * geo_loss
                        + args.consistency_weight * consistency_loss
                        + args.contrastive_weight * contrastive_loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals += [cell_loss.item(), country_loss.item(), offset_loss.item(),
                       geo_loss.item(), consistency_loss.item(), contrastive_loss.item()]
            n_batches += 1
        scheduler.step()
        row = {
            "epoch": epoch, "train_cell_loss": totals[0] / n_batches,
            "train_country_loss": totals[1] / n_batches, "train_offset_loss": totals[2] / n_batches,
            "train_geo_loss": totals[3] / n_batches, "train_consistency_loss": totals[4] / n_batches,
            "train_contrastive_loss": totals[5] / n_batches, "epoch_time_s": time.time() - started,
        }
        msg = (f"[full-data] {epoch + 1}/{args.epochs} cell={row['train_cell_loss']:.3f} "
               f"country={row['train_country_loss']:.3f} geo={row['train_geo_loss']:.3f} "
               f"{row['epoch_time_s']:.1f}s")
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            gallery = extract_gallery(model, gallery_loader, device)
            row["in_sample_median_km"] = in_sample_decode_check(model, gallery_loader, gallery, device)
            msg += f" in_sample(leaky)={row['in_sample_median_km']:.1f}km"
        history.append(row)
        print(msg, flush=True)
        torch.save(model.state_dict(), out_dir / "final_model.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    config = vars(args) | {
        "n_params_total": n_params, "param_limit": PARAM_LIMIT, "compliant": n_params <= PARAM_LIMIT,
        "preprocess": preprocess, "n_cells_per_partition": cell_info["n_cells"],
        "full_cells_dir": str(FULL_CELLS_DIR), "full_fold_id": FULL_FOLD_ID,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Artifacts written to {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
