"""Train the soft-hierarchical, geography-aware retrieval architecture.

The working T3.4 baseline is deliberately untouched.  This experiment keeps its country,
K-partition cell and offset supervision, adds a normalized geographic embedding, and
decodes against the labelled training gallery using the complete soft hierarchy rather
than hard country/cell argmaxes.
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
from torchvision import transforms

from src.data.dataset import GeoDataset, load_fold_cells_multi, load_labelled_df
from src.geo.retrieval import gallery_scores, spherical_topk_decode
from src.geo.spherical import (
    haversine_km_np,
    latlon_to_unit_vector_np,
    local_offset_km_to_latlon_np,
    unit_vector_to_latlon_np,
)
from src.models.retrieval_geo_classifier import RetrievalGeoClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "outputs" / "experiments"
PARAM_LIMIT = 5_000_000
OFFSET_SCALE_KM = 100.0


def build_transforms(backbone_name: str, input_size: int):
    probe = timm.create_model(backbone_name, pretrained=False)
    cfg = resolve_data_config({}, model=probe)
    del probe
    mean, std = cfg["mean"], cfg["std"]
    crop_pct = float(cfg.get("crop_pct", 0.9))
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                input_size, scale=(0.65, 1.0), ratio=(0.85, 1.15),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                round(input_size / crop_pct), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform, {
        "mean": list(mean), "std": list(std), "crop_pct": crop_pct,
    }


def make_loaders(
    val_fold: int, n_partitions: int, backbone_name: str, input_size: int,
    batch_size: int, num_workers: int,
):
    df = load_labelled_df()
    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]
    cell_info = load_fold_cells_multi(val_fold, n_partitions)
    train_tf, eval_tf, preprocess = build_transforms(backbone_name, input_size)

    train_ds = GeoDataset(
        train_df, train=True, input_size=input_size, multi_cell_info=cell_info
    )
    gallery_ds = GeoDataset(
        train_df, train=False, input_size=input_size, multi_cell_info=cell_info
    )
    val_ds = GeoDataset(
        val_df, train=False, input_size=input_size, multi_cell_info=cell_info
    )
    # New-pipeline-only override; the baseline dataset transform remains unchanged.
    train_ds.transform = train_tf
    gallery_ds.transform = eval_tf
    val_ds.transform = eval_tf

    common = dict(
        batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    gallery_loader = DataLoader(gallery_ds, shuffle=False, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, gallery_loader, val_loader, cell_info, preprocess


def masked_cell_loss(model, cell_logits_list, cell_ids, country_idx, sample_weight=None):
    """`sample_weight` (T4.1e cluster-upweight screen, 2026-08-20): optional per-sample
    multiplier, e.g. to upweight a confused-country subset (error_analysis_t4_1c.py found
    France/Germany/Italy/Poland/Spain broadly confusing each other, not just at borders --
    see border_analysis_t4_1c.py). `None` (default, every existing caller) is exactly the
    original unweighted per-partition mean."""
    losses = []
    for k, logits in enumerate(cell_logits_list):
        allowed = model.country_cell_mask(k)[country_idx]
        masked = logits.masked_fill(~allowed, -1e4)
        if sample_weight is None:
            losses.append(F.cross_entropy(masked, cell_ids[:, k]))
        else:
            per_sample = F.cross_entropy(masked, cell_ids[:, k], reduction="none")
            losses.append((per_sample * sample_weight).sum() / sample_weight.sum())
    return torch.stack(losses).mean()


def geo_distribution_loss(
    embeddings: torch.Tensor, lat: torch.Tensor, lng: torch.Tensor,
    geo_scale_km: float, embedding_temperature: float,
) -> torch.Tensor:
    """Match pairwise visual similarities to a smooth geographic neighbour distribution."""
    lat = torch.deg2rad(lat.float())
    lng = torch.deg2rad(lng.float())
    dlat = lat[:, None] - lat[None, :]
    dlng = lng[:, None] - lng[None, :]
    a = torch.sin(dlat / 2).square() + (
        torch.cos(lat[:, None]) * torch.cos(lat[None, :]) * torch.sin(dlng / 2).square()
    )
    distance = 2 * 6371.0088 * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))
    diagonal = torch.eye(len(lat), dtype=torch.bool, device=lat.device)
    target_logits = (-distance / geo_scale_km).masked_fill(diagonal, -1e4)
    target = F.softmax(target_logits, dim=-1)
    visual_logits = (
        embeddings.float() @ embeddings.float().T / embedding_temperature
    ).masked_fill(diagonal, -1e4)
    return -(target * F.log_softmax(visual_logits, dim=-1)).sum(dim=-1).mean()


def build_optimizer(model, backbone_lr: float, head_lr: float, weight_decay: float):
    head_params = (
        list(model.country_head.parameters()) + list(model.cls_heads.parameters())
        + list(model.offset_head.parameters()) + list(model.embedding_head.parameters())
    )
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


@torch.no_grad()
def extract_gallery(model, gallery_loader, device):
    model.eval()
    embeddings, countries, cell_ids, vectors = [], [], [], []
    for batch in gallery_loader:
        images = batch["image"].to(device, non_blocking=True)
        _, _, _, embedding = model(images)
        embeddings.append(embedding)
        countries.append(batch["country_idx"].to(device, non_blocking=True))
        cell_ids.append(batch["cell_ids"].to(device, non_blocking=True))
        vectors.append(batch["target_vec"].to(device, non_blocking=True))
    return {
        "embeddings": torch.cat(embeddings),
        "country": torch.cat(countries),
        "cell_ids": torch.cat(cell_ids),
        "unit_vectors": F.normalize(torch.cat(vectors), dim=-1),
    }


def hard_decode(model, cell_logits_list, offset_scaled, country_logits, centroids_list):
    pred_country = country_logits.argmax(dim=-1)
    pred_cells = []
    for k, logits in enumerate(cell_logits_list):
        allowed = model.country_cell_mask(k)[pred_country]
        pred_cells.append(logits.masked_fill(~allowed, -1e4).argmax(dim=-1).cpu().numpy())
    centroid_vectors = []
    for k, ids in enumerate(pred_cells):
        centroids = centroids_list[k][ids]
        centroid_vectors.append(latlon_to_unit_vector_np(centroids[:, 0], centroids[:, 1]))
    center_vec = np.mean(centroid_vectors, axis=0)
    center_lat, center_lng = unit_vector_to_latlon_np(center_vec)
    pred_lat, pred_lng = local_offset_km_to_latlon_np(
        offset_scaled.float().cpu().numpy() * OFFSET_SCALE_KM, center_lat, center_lng
    )
    return pred_lat, pred_lng, pred_country.cpu().numpy(), pred_cells


@torch.no_grad()
def evaluate(
    model, val_loader, gallery, centroids_list, device,
    retrieval_weights: list[float], top_ks: list[int], vote_temperature: float,
):
    model.eval()
    decoder_errors = {
        f"retrieval_w{weight:g}_k{top_k}": []
        for weight in retrieval_weights for top_k in top_ks
    }
    hard_errors = []
    n_country_correct = 0
    n_cell_correct = np.zeros(len(centroids_list), dtype=np.int64)
    n_total = 0
    masks = [model.country_cell_mask(k) for k in range(model.n_partitions)]

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        cell_logits, offset, country_logits, embeddings = model(images)
        hard_lat, hard_lng, pred_country, pred_cells = hard_decode(
            model, cell_logits, offset, country_logits, centroids_list
        )
        true_lat = batch["lat"].numpy()
        true_lng = batch["lng"].numpy()
        hard_errors.extend(haversine_km_np(true_lat, true_lng, hard_lat, hard_lng).tolist())

        true_country = batch["country_idx"].numpy()
        true_cells = batch["cell_ids"].numpy()
        n_country_correct += int((pred_country == true_country).sum())
        for k in range(model.n_partitions):
            n_cell_correct[k] += int((pred_cells[k] == true_cells[:, k]).sum())
        n_total += len(true_country)

        for weight in retrieval_weights:
            scores = gallery_scores(
                country_logits, cell_logits, embeddings, masks,
                gallery["country"], gallery["cell_ids"], gallery["embeddings"], weight,
            )
            for top_k in top_ks:
                pred_lat, pred_lng, _ = spherical_topk_decode(
                    scores, gallery["unit_vectors"], top_k, vote_temperature
                )
                key = f"retrieval_w{weight:g}_k{top_k}"
                decoder_errors[key].extend(
                    haversine_km_np(
                        true_lat, true_lng,
                        pred_lat.cpu().numpy(), pred_lng.cpu().numpy(),
                    ).tolist()
                )

    decoder_stats = {}
    for key, errors in decoder_errors.items():
        errors_np = np.asarray(errors)
        decoder_stats[key] = {
            "median_km": float(np.median(errors_np)),
            "mean_km": float(np.mean(errors_np)),
            "frac_lt_200km": float(np.mean(errors_np < 200)),
            "frac_lt_750km": float(np.mean(errors_np < 750)),
        }
    hard_np = np.asarray(hard_errors)
    best_decoder = min(decoder_stats, key=lambda key: decoder_stats[key]["median_km"])
    return {
        "median_km": decoder_stats[best_decoder]["median_km"],
        "mean_km": decoder_stats[best_decoder]["mean_km"],
        "frac_lt_200km": decoder_stats[best_decoder]["frac_lt_200km"],
        "frac_lt_750km": decoder_stats[best_decoder]["frac_lt_750km"],
        "best_decoder": best_decoder,
        "decoder_stats": decoder_stats,
        "hard_median_km": float(np.median(hard_np)),
        "hard_mean_km": float(np.mean(hard_np)),
        "country_top1_acc": n_country_correct / n_total,
        "cell_top1_acc_per_partition": (n_cell_correct / n_total).tolist(),
        "cell_top1_acc": float(np.mean(n_cell_correct / n_total)),
    }


def train_one_fold(args, val_fold: int, device: torch.device, out_dir: Path):
    train_loader, gallery_loader, val_loader, cell_info, preprocess = make_loaders(
        val_fold, args.n_partitions, args.backbone, args.input_size,
        args.batch_size, args.num_workers,
    )
    model = RetrievalGeoClassifier(
        args.backbone, cell_info["n_cells"], cell_info["cell_to_country"],
        embedding_dim=args.embedding_dim, pretrained=True, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > PARAM_LIMIT:
        raise RuntimeError(f"model has {n_params:,} parameters; limit is {PARAM_LIMIT:,}")
    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_median = float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sums = np.zeros(4, dtype=np.float64)
        n_batches = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            cells = batch["cell_ids"].to(device, non_blocking=True)
            countries = batch["country_idx"].to(device, non_blocking=True)
            offsets = batch["offset_km"].to(device, non_blocking=True)
            lat = batch["lat"].to(device, non_blocking=True)
            lng = batch["lng"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                cell_logits, offset_pred, country_logits, embedding = model(images)
                cell_loss = masked_cell_loss(model, cell_logits, cells, countries)
                country_loss = F.cross_entropy(country_logits, countries)
                offset_loss = F.huber_loss(
                    offset_pred, offsets / OFFSET_SCALE_KM, delta=1.0
                )
                geo_loss = geo_distribution_loss(
                    embedding, lat, lng, args.geo_scale_km, args.embedding_temperature
                )
                loss = (
                    cell_loss + args.country_weight * country_loss
                    + args.offset_weight * offset_loss + args.geo_weight * geo_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sums += [cell_loss.item(), country_loss.item(), offset_loss.item(), geo_loss.item()]
            n_batches += 1
        scheduler.step()

        gallery = extract_gallery(model, gallery_loader, device)
        stats = evaluate(
            model, val_loader, gallery, cell_info["centroids"], device,
            args.retrieval_weights, args.top_ks, args.vote_temperature,
        )
        elapsed = time.time() - t0
        row = {
            "epoch": epoch, "train_cell_loss": loss_sums[0] / n_batches,
            "train_country_loss": loss_sums[1] / n_batches,
            "train_offset_loss": loss_sums[2] / n_batches,
            "train_geo_loss": loss_sums[3] / n_batches,
            "epoch_time_s": elapsed, **stats,
        }
        history.append(row)
        print(
            f"[fold {val_fold}] {epoch+1}/{args.epochs} "
            f"losses={row['train_cell_loss']:.3f}/{row['train_country_loss']:.3f}/"
            f"{row['train_geo_loss']:.3f} country={100*stats['country_top1_acc']:.1f}% "
            f"cell={100*stats['cell_top1_acc']:.1f}% hard={stats['hard_median_km']:.1f}km "
            f"best={stats['median_km']:.1f}km ({stats['best_decoder']}) {elapsed:.1f}s"
        )
        if stats["median_km"] < best_median:
            best_median = stats["median_km"]
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    gallery = extract_gallery(model, gallery_loader, device)
    best_stats = evaluate(
        model, val_loader, gallery, cell_info["centroids"], device,
        args.retrieval_weights, args.top_ks, args.vote_temperature,
    )
    result = {
        "best_epoch": best_epoch, "best_val_median_km": best_stats["median_km"],
        "n_params_total": n_params, "preprocess": preprocess, **best_stats,
    }
    (out_dir / "fold_result.json").write_text(json.dumps(result, indent=2))
    return result, n_params, preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="efficientnet_b0.ra4_e3600_r224_in1k")
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--n-partitions", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--country-weight", type=float, default=1.0)
    parser.add_argument("--offset-weight", type=float, default=0.5)
    parser.add_argument("--geo-weight", type=float, default=0.2)
    parser.add_argument("--geo-scale-km", type=float, default=250.0)
    parser.add_argument("--embedding-temperature", type=float, default=0.1)
    parser.add_argument("--retrieval-weights", type=float, nargs="+", default=[0.0, 2.0, 4.0, 8.0])
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--vote-temperature", type=float, default=1.0)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--exp-name", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = OUT_DIR_BASE / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    fold_results = {}
    n_params = None
    preprocess = None
    for fold in args.folds:
        result, n_params, preprocess = train_one_fold(
            args, fold, device, exp_dir / f"fold{fold}"
        )
        fold_results[fold] = result
    config = vars(args) | {
        "n_params_total": n_params, "param_limit": PARAM_LIMIT,
        "compliant": n_params <= PARAM_LIMIT, "preprocess": preprocess,
    }
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2))
    (exp_dir / "param_compliance.json").write_text(json.dumps(
        {k: config[k] for k in ["n_params_total", "param_limit", "compliant"]}, indent=2
    ))
    if len(fold_results) > 1:
        medians = np.array([r["best_val_median_km"] for r in fold_results.values()])
        summary = {
            "backbone": args.backbone, "folds_run": args.folds,
            "cv_median_km_mean": float(medians.mean()),
            "cv_median_km_std": float(medians.std(ddof=1)),
            "per_fold": {str(k): v for k, v in fold_results.items()},
        }
        (exp_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Artifacts written to {exp_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
