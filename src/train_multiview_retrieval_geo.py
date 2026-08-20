"""Train T4.1b: shared global/local views with mode-aware geographic retrieval."""

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
from src.geo.mode_retrieval import spherical_mode_decode
from src.geo.retrieval import conditional_cell_log_probs, spherical_topk_decode
from src.geo.spherical import haversine_km_np
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier
from src.train_retrieval_geo import (
    PARAM_LIMIT,
    build_optimizer,
    geo_distribution_loss,
    hard_decode,
    masked_cell_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "outputs" / "experiments"


def make_multiview_loaders(args, val_fold: int):
    df = load_labelled_df()
    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]
    cell_info = load_fold_cells_multi(val_fold, args.n_partitions)
    probe = timm.create_model(args.backbone, pretrained=False)
    cfg = resolve_data_config({}, model=probe)
    del probe
    preprocess = {
        "mean": list(cfg["mean"]), "std": list(cfg["std"]),
        "crop_pct": float(cfg.get("crop_pct", 0.9)),
        "views": "global_plus_four_corner_crops",
        "local_fraction": args.local_fraction,
    }
    train_ds = GeoDataset(
        train_df, train=True, input_size=args.input_size, multi_cell_info=cell_info
    )
    gallery_ds = GeoDataset(
        train_df, train=False, input_size=args.input_size, multi_cell_info=cell_info
    )
    val_ds = GeoDataset(
        val_df, train=False, input_size=args.input_size, multi_cell_info=cell_info
    )
    train_ds.transform = GlobalLocalTransform(
        args.input_size, cfg["mean"], cfg["std"], preprocess["crop_pct"], True,
        args.local_fraction,
    )
    eval_transform = GlobalLocalTransform(
        args.input_size, cfg["mean"], cfg["std"], preprocess["crop_pct"], False,
        args.local_fraction,
    )
    gallery_ds.transform = eval_transform
    val_ds.transform = eval_transform
    sampler = GeoBalancedBatchSampler(
        train_ds.df["filename"].tolist(), cell_info["assignments"][0],
        args.batch_size, args.samples_per_cell,
    )
    worker_args = dict(
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **worker_args)
    eval_args = dict(batch_size=args.eval_batch_size, shuffle=False, drop_last=False, **worker_args)
    return (
        train_loader, DataLoader(gallery_ds, **eval_args), DataLoader(val_ds, **eval_args),
        cell_info, preprocess,
    )


@torch.no_grad()
def extract_gallery(model, loader, device):
    model.eval()
    out = {key: [] for key in ["embeddings", "local_embeddings", "country", "cell_ids", "unit_vectors"]}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        _, _, _, embedding, views = model(images)
        out["embeddings"].append(embedding)
        out["local_embeddings"].append(F.normalize(views[:, 1:].mean(dim=1), dim=-1))
        out["country"].append(batch["country_idx"].to(device, non_blocking=True))
        out["cell_ids"].append(batch["cell_ids"].to(device, non_blocking=True))
        out["unit_vectors"].append(batch["target_vec"].to(device, non_blocking=True))
    return {key: F.normalize(torch.cat(value), dim=-1) if key == "unit_vectors" else torch.cat(value)
            for key, value in out.items()}


def hierarchy_scores(model, country_logits, cell_logits, gallery):
    scores = F.log_softmax(country_logits.float(), dim=-1)[:, gallery["country"]]
    for k, logits in enumerate(cell_logits):
        conditional = conditional_cell_log_probs(logits.float(), model.country_cell_mask(k))
        scores = scores + conditional[:, gallery["cell_ids"][:, k]] / model.n_partitions
    return scores


def supervised_geo_contrastive_loss(embeddings, cell_ids, temperature):
    """Pull together samples sharing geographic cells while retaining hard negatives."""
    diagonal = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    similarity = embeddings.float() @ embeddings.float().T / temperature
    log_prob = F.log_softmax(similarity.masked_fill(diagonal, -1e4), dim=-1)
    positive_weight = (cell_ids[:, None, :] == cell_ids[None, :, :]).float().mean(dim=-1)
    positive_weight = positive_weight.masked_fill(diagonal, 0.0)
    positive_weight = positive_weight / positive_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(positive_weight * log_prob).sum(dim=-1).mean()


@torch.no_grad()
def evaluate(model, loader, gallery, centroids, device, args):
    model.eval()
    specs = [(w, lw, radius) for w in args.retrieval_weights
             for lw in args.local_weights for radius in args.mode_radii_km]
    errors = {f"mode_w{w:g}_l{lw:g}_r{radius:g}": [] for w, lw, radius in specs}
    errors.update({f"plain_w{w:g}_l{lw:g}_k10": [] for w in args.retrieval_weights
                   for lw in args.local_weights})
    hard_errors, n_country_correct, n_total = [], 0, 0
    n_cell_correct = np.zeros(model.n_partitions, dtype=np.int64)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        cell_logits, offset, country_logits, embedding, views = model(images)
        hard_lat, hard_lng, pred_country, pred_cells = hard_decode(
            model, cell_logits, offset, country_logits, centroids
        )
        true_lat, true_lng = batch["lat"].numpy(), batch["lng"].numpy()
        hard_errors.extend(haversine_km_np(true_lat, true_lng, hard_lat, hard_lng))
        true_country, true_cells = batch["country_idx"].numpy(), batch["cell_ids"].numpy()
        n_country_correct += int((pred_country == true_country).sum())
        for k in range(model.n_partitions):
            n_cell_correct[k] += int((pred_cells[k] == true_cells[:, k]).sum())
        n_total += len(true_country)

        hierarchy = hierarchy_scores(model, country_logits, cell_logits, gallery)
        fused_similarity = embedding.float() @ gallery["embeddings"].float().T
        local_similarity = torch.einsum(
            "bvd,nd->bvn", views[:, 1:].float(), gallery["local_embeddings"].float()
        ).amax(dim=1)
        for weight in args.retrieval_weights:
            for local_weight in args.local_weights:
                similarity = (1 - local_weight) * fused_similarity + local_weight * local_similarity
                scores = hierarchy + weight * similarity
                pred_lat, pred_lng, _ = spherical_topk_decode(
                    scores, gallery["unit_vectors"], top_k=10
                )
                key = f"plain_w{weight:g}_l{local_weight:g}_k10"
                errors[key].extend(haversine_km_np(
                    true_lat, true_lng, pred_lat.cpu().numpy(), pred_lng.cpu().numpy()
                ))
                for radius in args.mode_radii_km:
                    pred_lat, pred_lng, _ = spherical_mode_decode(
                        scores, gallery["unit_vectors"], top_m=args.mode_top_m,
                        radius_km=radius, kernel=args.mode_kernel,
                    )
                    key = f"mode_w{weight:g}_l{local_weight:g}_r{radius:g}"
                    errors[key].extend(haversine_km_np(
                        true_lat, true_lng, pred_lat.cpu().numpy(), pred_lng.cpu().numpy()
                    ))
    decoder_stats = {}
    for key, values in errors.items():
        values = np.asarray(values)
        decoder_stats[key] = {
            "median_km": float(np.median(values)), "mean_km": float(values.mean()),
            "frac_lt_200km": float(np.mean(values < 200)),
            "frac_lt_750km": float(np.mean(values < 750)),
        }
    best = min(decoder_stats, key=lambda key: decoder_stats[key]["median_km"])
    hard_errors = np.asarray(hard_errors)
    return {
        **decoder_stats[best], "best_decoder": best, "decoder_stats": decoder_stats,
        "hard_median_km": float(np.median(hard_errors)),
        "hard_mean_km": float(hard_errors.mean()),
        "country_top1_acc": n_country_correct / n_total,
        "cell_top1_acc_per_partition": (n_cell_correct / n_total).tolist(),
        "cell_top1_acc": float(np.mean(n_cell_correct / n_total)),
    }


def train_one_fold(args, fold, device, out_dir):
    train_loader, gallery_loader, val_loader, cells, preprocess = make_multiview_loaders(args, fold)
    model = MultiViewRetrievalGeoClassifier(
        args.backbone, cells["n_cells"], cells["cell_to_country"],
        embedding_dim=args.embedding_dim, pretrained=not bool(args.init_checkpoint),
        dropout=args.dropout,
    ).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"Initialized from {args.init_checkpoint}", flush=True)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    if n_params > PARAM_LIMIT:
        raise RuntimeError(f"model has {n_params:,} parameters; limit is {PARAM_LIMIT:,}")
    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_median, best_epoch = [], float("inf"), -1
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
        gallery = extract_gallery(model, gallery_loader, device)
        stats = evaluate(model, val_loader, gallery, cells["centroids"], device, args)
        row = {"epoch": epoch, "train_cell_loss": totals[0] / n_batches,
               "train_country_loss": totals[1] / n_batches,
               "train_offset_loss": totals[2] / n_batches,
               "train_geo_loss": totals[3] / n_batches,
               "train_consistency_loss": totals[4] / n_batches,
               "train_contrastive_loss": totals[5] / n_batches,
               "epoch_time_s": time.time() - started, **stats}
        history.append(row)
        print(f"[fold {fold}] {epoch+1}/{args.epochs} cell={row['train_cell_loss']:.3f} "
              f"geo={row['train_geo_loss']:.3f} consistency={row['train_consistency_loss']:.3f} "
              f"hard={stats['hard_median_km']:.1f}km best={stats['median_km']:.1f}km "
              f"({stats['best_decoder']}) {row['epoch_time_s']:.1f}s", flush=True)
        if stats["median_km"] < best_median:
            best_median, best_epoch = stats["median_km"], epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    best_stats = evaluate(model, val_loader, extract_gallery(model, gallery_loader, device),
                          cells["centroids"], device, args)
    result = {"best_epoch": best_epoch, "best_val_median_km": best_stats["median_km"],
              "n_params_total": n_params, "preprocess": preprocess, **best_stats}
    (out_dir / "fold_result.json").write_text(json.dumps(result, indent=2))
    return result, n_params, preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0.ra4_e3600_r224_in1k")
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--n-partitions", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--samples-per-cell", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--country-weight", type=float, default=1.0)
    parser.add_argument("--offset-weight", type=float, default=0.5)
    parser.add_argument("--geo-weight", type=float, default=0.2)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--geo-scale-km", type=float, default=250.0)
    parser.add_argument("--embedding-temperature", type=float, default=0.1)
    parser.add_argument("--retrieval-weights", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--local-weights", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--mode-radii-km", type=float, nargs="+", default=[150.0, 250.0])
    parser.add_argument("--mode-top-m", type=int, default=50)
    parser.add_argument("--mode-kernel", choices=["gaussian", "hard"], default="hard")
    parser.add_argument("--local-fraction", type=float, default=0.64)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--exp-name", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = OUT_DIR_BASE / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    results, n_params, preprocess = {}, None, None
    for fold in args.folds:
        results[fold], n_params, preprocess = train_one_fold(
            args, fold, device, exp_dir / f"fold{fold}"
        )
    config = vars(args) | {"n_params_total": n_params, "param_limit": PARAM_LIMIT,
                           "compliant": n_params <= PARAM_LIMIT, "preprocess": preprocess}
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "param_compliance.json").write_text(json.dumps(
        {key: config[key] for key in ["n_params_total", "param_limit", "compliant"]}, indent=2
    ))
    print(f"Artifacts written to {exp_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
