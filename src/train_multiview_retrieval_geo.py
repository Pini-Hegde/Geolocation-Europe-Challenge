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

from src.data.dataset import (
    GeoDataset,
    load_fold_cells_multi,
    load_fold_region,
    load_labelled_df,
)
from src.data.multiview import GeoBalancedBatchSampler, GlobalLocalTransform
from src.geo.countries import COUNTRY_TO_IDX
from src.geo.mode_retrieval import spherical_mode_decode
from src.geo.retrieval import conditional_cell_log_probs, spherical_topk_decode
from src.geo.spherical import haversine_km_np, latlon_to_unit_vector_np
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier
from src.train_retrieval_geo import (
    PARAM_LIMIT,
    build_optimizer,
    geo_distribution_loss,
    hard_decode,
    haversine_smoothed_cell_loss,
    masked_cell_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "outputs" / "experiments"


def make_multiview_loaders(args, val_fold: int):
    df = load_labelled_df()
    if val_fold < 0:  # full-data: train on everything, tiny dummy val = fold 0 (leaky, ignored)
        train_df = df
        val_df = df[df["fold"] == 0].iloc[:64]
    else:
        train_df = df[df["fold"] != val_fold]
        val_df = df[df["fold"] == val_fold]
    cells_dir = Path(args.cells_dir) if args.cells_dir else None
    cell_info = load_fold_cells_multi(val_fold, args.n_partitions, cells_dir=cells_dir)
    cell_info["train_coords"] = (
        torch.tensor(train_df["lat"].to_numpy(), dtype=torch.float32),
        torch.tensor(train_df["lng"].to_numpy(), dtype=torch.float32),
    )
    cell_info["centroid_vecs"] = [
        torch.as_tensor(
            latlon_to_unit_vector_np(c[:, 0], c[:, 1]), dtype=torch.float32
        )
        for c in cell_info["centroids"]
    ]
    region_info = None
    if getattr(args, "region_weight", 0.0) > 0:
        region_info = load_fold_region(
            val_fold, region_dir=Path(args.cells_dir) if args.cells_dir else None
        )
    cell_info["region_info"] = region_info
    probe = timm.create_model(args.backbone, pretrained=False)
    cfg = resolve_data_config({}, model=probe)
    del probe
    preprocess = {
        "mean": list(cfg["mean"]), "std": list(cfg["std"]),
        "crop_pct": float(cfg.get("crop_pct", 0.9)),
        "views": "global_plus_four_corner_crops",
        "local_fraction": args.local_fraction,
    }
    fix_o = bool(getattr(args, "fix_orientation", False))
    train_ds = GeoDataset(
        train_df, train=True, input_size=args.input_size, multi_cell_info=cell_info,
        region_info=region_info, fix_orientation=fix_o,
    )
    gallery_ds = GeoDataset(
        train_df, train=False, input_size=args.input_size, multi_cell_info=cell_info,
        region_info=region_info, fix_orientation=fix_o,
    )
    val_ds = GeoDataset(
        val_df, train=False, input_size=args.input_size, multi_cell_info=cell_info,
        region_info=region_info, fix_orientation=fix_o,
    )
    train_ds.transform = GlobalLocalTransform(
        args.input_size, cfg["mean"], cfg["std"], preprocess["crop_pct"], True,
        args.local_fraction, strong_aug=bool(getattr(args, "strong_aug", False)),
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
        seed=1337 + int(getattr(args, "seed", 0)),
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
    has_loc = getattr(model, "use_location_encoder", False)
    has_region = getattr(model, "n_regions", 0) > 0
    keys = ["embeddings", "local_embeddings", "country", "cell_ids", "unit_vectors"]
    if has_loc:
        keys.append("loc_embeddings")
    if has_region:
        keys.append("region_ids")
    out = {key: [] for key in keys}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        _, _, _, embedding, views = model(images)
        out["embeddings"].append(embedding)
        out["local_embeddings"].append(F.normalize(views[:, 1:].mean(dim=1), dim=-1))
        out["country"].append(batch["country_idx"].to(device, non_blocking=True))
        out["cell_ids"].append(batch["cell_ids"].to(device, non_blocking=True))
        out["unit_vectors"].append(batch["target_vec"].to(device, non_blocking=True))
        if has_region:
            out["region_ids"].append(batch["region_id"].to(device, non_blocking=True))
        if has_loc:
            out["loc_embeddings"].append(
                model.encode_location(
                    batch["lat"].float().to(device), batch["lng"].float().to(device)
                )
            )
    return {key: F.normalize(torch.cat(value), dim=-1) if key == "unit_vectors" else torch.cat(value)
            for key, value in out.items()}


def hierarchy_scores(model, country_logits, cell_logits, gallery, region_logits=None,
                     region_weight=1.0):
    scores = F.log_softmax(country_logits.float(), dim=-1)[:, gallery["country"]]
    for k, logits in enumerate(cell_logits):
        conditional = conditional_cell_log_probs(logits.float(), model.country_cell_mask(k))
        scores = scores + conditional[:, gallery["cell_ids"][:, k]] / model.n_partitions
    if region_logits is not None and "region_ids" in gallery and getattr(model, "n_regions", 0):
        rcond = conditional_cell_log_probs(region_logits.float(), model.country_region_mask)
        scores = scores + region_weight * rcond[:, gallery["region_ids"]]
    return scores


def location_alignment_loss(
    img_emb, model, lat, lng, train_lat, train_lng, temperature, n_neg=384
):
    """GeoCLIP-style (Cepeda et al. 2023, Eq. 4) InfoNCE between the image embedding and the
    location embedding of its true GPS. In-batch negatives + `n_neg` synthetic negatives
    drawn (with jitter) from the training-set coordinate pool -- so the location encoder
    learns a smooth field over the sampled region, not just the batch's points."""
    from src.models.location_encoder import sample_synthetic_coords

    loc_emb = model.encode_location(lat.float(), lng.float())  # (B, D)
    logits_i2l = img_emb.float() @ loc_emb.T / temperature  # (B, B)
    labels = torch.arange(len(img_emb), device=img_emb.device)
    neg_lat, neg_lng = sample_synthetic_coords(train_lat, train_lng, n_neg, jitter_km=25.0)
    neg_emb = model.encode_location(
        neg_lat.to(img_emb.device), neg_lng.to(img_emb.device)
    )
    logits_i2neg = img_emb.float() @ neg_emb.T / temperature  # (B, n_neg)
    full = torch.cat([logits_i2l, logits_i2neg], dim=1)
    loss_i = F.cross_entropy(full, labels)
    loss_l = F.cross_entropy(logits_i2l.T, labels)
    return 0.5 * (loss_i + loss_l)


def supervised_geo_contrastive_loss(
    embeddings, cell_ids, temperature, country_idx=None, country_blend: float = 0.0
):
    """Pull together samples sharing geographic cells while retaining hard negatives.

    `country_blend` (T6/E-contrastive, 2026-08-27): OSV-5M (Astruc et al. 2024, Tab. 5)
    finds contrastive positives defined at the *region/country* level beat cell/area-level
    positives -- coarser divisions carry more unique, homogeneous visual identity. With
    `country_blend > 0` the positive weight becomes
    `(1 - blend) * shared_cell_fraction + blend * same_country`.
    """
    diagonal = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    similarity = embeddings.float() @ embeddings.float().T / temperature
    log_prob = F.log_softmax(similarity.masked_fill(diagonal, -1e4), dim=-1)
    positive_weight = (cell_ids[:, None, :] == cell_ids[None, :, :]).float().mean(dim=-1)
    if country_blend > 0 and country_idx is not None:
        same_country = (country_idx[:, None] == country_idx[None, :]).float()
        positive_weight = (1 - country_blend) * positive_weight + country_blend * same_country
    positive_weight = positive_weight.masked_fill(diagonal, 0.0)
    positive_weight = positive_weight / positive_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(positive_weight * log_prob).sum(dim=-1).mean()


def hard_negative_geo_loss(embeddings, lat, lng, far_km=400.0, margin=0.25):
    """T6/E9: explicitly push apart the embeddings of image pairs that are VISUALLY similar
    (high cosine) but GEOGRAPHICALLY far (> far_km) -- the France<->Germany rural-road
    confusion. For each anchor, penalise its single most-similar geographically-far partner
    with a hinge on cos_sim - margin. This forces the model to find *some* discriminating
    feature (road markings, poles, vegetation) rather than collapsing featureless scenes
    from different countries onto the same point."""
    cos = F.normalize(embeddings.float(), dim=-1) @ F.normalize(embeddings.float(), dim=-1).T
    lat_r = torch.deg2rad(lat.float()); lng_r = torch.deg2rad(lng.float())
    dlat = lat_r[:, None] - lat_r[None, :]
    dlng = lng_r[:, None] - lng_r[None, :]
    a = torch.sin(dlat / 2).square() + (
        torch.cos(lat_r[:, None]) * torch.cos(lat_r[None, :]) * torch.sin(dlng / 2).square())
    dist_km = 2 * 6371.0088 * torch.asin(torch.sqrt(a.clamp(0, 1)))
    far = dist_km > far_km
    cos_far = cos.masked_fill(~far, -2.0)
    hardest = cos_far.max(dim=-1).values  # most-similar far partner per anchor
    return F.relu(hardest - margin).mean()


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
        region_logits = model.last_region_logits
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

        hierarchy = hierarchy_scores(model, country_logits, cell_logits, gallery,
                                     region_logits=region_logits,
                                     region_weight=float(getattr(args, "region_decode_weight", 1.0)))
        fused_similarity = embedding.float() @ gallery["embeddings"].float().T
        loc_decode_weight = float(getattr(args, "loc_decode_weight", 0.0))
        if loc_decode_weight > 0 and "loc_embeddings" in gallery:
            fused_similarity = fused_similarity + loc_decode_weight * (
                embedding.float() @ gallery["loc_embeddings"].float().T
            )
        if any(lw > 0 for lw in args.local_weights):
            local_similarity = torch.einsum(
                "bvd,nd->bvn", views[:, 1:].float(), gallery["local_embeddings"].float()
            ).amax(dim=1)
        else:
            local_similarity = torch.zeros_like(fused_similarity)
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
        cell_proj_dim=(args.cell_proj_dim if args.cell_proj_dim > 0 else None),
        use_location_encoder=(float(getattr(args, "loc_align_weight", 0.0)) > 0),
        country_head_hidden=int(getattr(args, "country_head_hidden", 0)),
        region_to_country=(cells["region_info"]["region_to_country"]
                           if cells.get("region_info") is not None else None),
    ).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device), strict=False)
        print(f"Initialized from {args.init_checkpoint}", flush=True)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    if n_params > PARAM_LIMIT:
        raise RuntimeError(f"model has {n_params:,} parameters; limit is {PARAM_LIMIT:,}")
    optimizer = build_optimizer(model, args.backbone_lr, args.head_lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    cluster_country_idx = torch.tensor(
        [COUNTRY_TO_IDX[c] for c in args.cluster_countries], device=device
    )
    distill_probs = None
    if getattr(args, "distill_country_npz", ""):
        d = np.load(args.distill_country_npz, allow_pickle=True)
        distill_probs = {fn: torch.tensor(p, device=device)
                         for fn, p in zip(d["filenames"].tolist(), d["probs"])}
        print(f"country distillation: {len(distill_probs)} teacher targets, "
              f"weight={args.distill_weight}", flush=True)
    tau_km = float(getattr(args, "label_smooth_tau", 0.0))
    centroid_vecs = [v.to(device) for v in cells["centroid_vecs"]] if tau_km > 0 else None
    ema = None
    if float(getattr(args, "ema_decay", 0.0)) > 0:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=args.ema_decay, use_warmup=True, device=device)
        print(f"EMA enabled (decay={args.ema_decay})", flush=True)
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
            target_vec = batch["target_vec"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            sample_weight = None
            if args.cluster_weight != 1.0:
                sample_weight = torch.where(
                    (countries[:, None] == cluster_country_idx[None, :]).any(dim=-1),
                    args.cluster_weight, 1.0,
                )
            region_true = (batch["region_id"].to(device, non_blocking=True)
                           if "region_id" in batch else None)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, offset, country_logits, embedding, views = model(images)
                region_logits = model.last_region_logits
                if region_logits is not None and region_true is not None:
                    rallowed = model.country_region_mask[countries]
                    region_loss = F.cross_entropy(
                        region_logits.masked_fill(~rallowed, -1e4), region_true)
                else:
                    region_loss = torch.zeros((), device=device)
                if centroid_vecs is not None:
                    cell_loss = haversine_smoothed_cell_loss(
                        model, logits, countries, target_vec, centroid_vecs, tau_km,
                        sample_weight,
                    )
                else:
                    cell_loss = masked_cell_loss(
                        model, logits, cells_true, countries, sample_weight
                    )
                if sample_weight is None:
                    country_loss = F.cross_entropy(country_logits, countries)
                else:
                    country_loss = (
                        F.cross_entropy(country_logits, countries, reduction="none") * sample_weight
                    ).sum() / sample_weight.sum()
                offset_loss = F.huber_loss(offset, offsets / 100.0, delta=1.0)
                geo_loss = geo_distribution_loss(
                    embedding, lat, lng, args.geo_scale_km, args.embedding_temperature
                )
                consistency_loss = (1 - (views * embedding[:, None]).sum(dim=-1)).mean()
                contrastive_loss = supervised_geo_contrastive_loss(
                    embedding, cells_true, args.contrastive_temperature,
                    country_idx=countries,
                    country_blend=float(getattr(args, "contrastive_country_blend", 0.0)),
                )
                loc_align_weight = float(getattr(args, "loc_align_weight", 0.0))
                if loc_align_weight > 0:
                    loc_loss = location_alignment_loss(
                        embedding, model, lat, lng,
                        cells["train_coords"][0], cells["train_coords"][1],
                        args.loc_temperature,
                    )
                else:
                    loc_loss = torch.zeros((), device=device)
                hn_weight = float(getattr(args, "hard_neg_weight", 0.0))
                if hn_weight > 0:
                    hn_loss = hard_negative_geo_loss(
                        embedding, lat, lng,
                        far_km=float(getattr(args, "hard_neg_far_km", 400.0)),
                        margin=float(getattr(args, "hard_neg_margin", 0.25)))
                else:
                    hn_loss = torch.zeros((), device=device)
                if distill_probs is not None:
                    teacher_p = torch.stack([distill_probs[fn] for fn in batch["filename"]])
                    distill_loss = F.kl_div(
                        F.log_softmax(country_logits.float(), dim=-1),
                        teacher_p.float(), reduction="batchmean")
                else:
                    distill_loss = torch.zeros((), device=device)
                loss = (cell_loss + args.country_weight * country_loss
                        + float(getattr(args, "distill_weight", 0.0)) * distill_loss
                        + args.offset_weight * offset_loss + args.geo_weight * geo_loss
                        + args.consistency_weight * consistency_loss
                        + args.contrastive_weight * contrastive_loss
                        + loc_align_weight * loc_loss
                        + float(getattr(args, "region_weight", 0.0)) * region_loss
                        + hn_weight * hn_loss)
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            totals += [cell_loss.item(), country_loss.item(), offset_loss.item(),
                       geo_loss.item(), consistency_loss.item(), contrastive_loss.item()]
            n_batches += 1
        scheduler.step()
        full_data = fold < 0
        if full_data:
            # no honest val -> just checkpoint every epoch, keep the LAST (E3e converged ~ep52/55)
            torch.save((ema.module if ema is not None else model).state_dict(),
                       out_dir / "best_model.pt")
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
            print(f"[full-data] {epoch+1}/{args.epochs} cell={totals[0]/n_batches:.3f} "
                  f"geo={totals[3]/n_batches:.3f} {time.time()-started:.1f}s", flush=True)
            history.append({"epoch": epoch, "train_cell_loss": totals[0]/n_batches,
                            "epoch_time_s": time.time()-started})
            continue
        eval_every = max(1, getattr(args, "eval_every", 1))
        do_eval = ((epoch + 1) % eval_every == 0) or (epoch >= args.epochs - 3) or (epoch == 0)
        if not do_eval:
            print(f"[fold {fold}] {epoch+1}/{args.epochs} cell={totals[0]/n_batches:.3f} "
                  f"(eval skipped) {time.time()-started:.1f}s", flush=True)
            continue
        eval_model = ema.module if ema is not None else model
        gallery = extract_gallery(eval_model, gallery_loader, device)
        stats = evaluate(eval_model, val_loader, gallery, cells["centroids"], device, args)
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
            torch.save(eval_model.state_dict(), out_dir / "best_model.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    final_model = ema.module if ema is not None else model
    if fold < 0:
        result = {"best_epoch": args.epochs - 1, "n_params_total": n_params,
                  "preprocess": preprocess, "full_data": True}
        (out_dir / "fold_result.json").write_text(json.dumps(result, indent=2))
        return result, n_params, preprocess
    final_model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    best_stats = evaluate(final_model, val_loader,
                          extract_gallery(final_model, gallery_loader, device),
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
    parser.add_argument(
        "--cells-dir", default="",
        help="Override for outputs/protocol/geo_cells_multi (screening alt cell-size caches); "
             "empty = canonical directory.",
    )
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument(
        "--country-head-hidden", type=int, default=0,
        help="T6/E3g: hidden dim of an MLP country head (0 = single Linear). ~128-256.",
    )
    parser.add_argument(
        "--cell-proj-dim", type=int, default=0,
        help="T6/E3+: shared low-rank projection dim before the K cell heads (0 = off, "
             "direct 1280->cells). Use ~192-256 to fit many fine cells under 5M params.",
    )
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
    parser.add_argument(
        "--grad-clip-norm", type=float, default=0.0,
        help="Max gradient L2 norm (torch.nn.utils.clip_grad_norm_), applied AMP-safely via "
             "scaler.unscale_ before clipping. 0 (default) = disabled, identical to every "
             "prior run in this project -- none of them clip gradients.",
    )
    parser.add_argument(
        "--cluster-weight", type=float, default=1.0,
        help="Per-sample cell/country loss multiplier for --cluster-countries (T4.1e, "
             "2026-08-20 -- error_analysis_t4_1c.py found France/Germany/Italy/Poland/Spain "
             "broadly confusing each other; border_analysis_t4_1c.py ruled out a border-only "
             "effect). 1.0 = no-op, identical to every prior run.",
    )
    parser.add_argument(
        "--cluster-countries", nargs="+",
        default=["France", "Germany", "Italy", "Poland", "Spain"],
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.0,
        help="T6/E5: EMA of weights for eval + checkpoint (0 = off). Try 0.998-0.9995.",
    )
    parser.add_argument(
        "--label-smooth-tau", type=float, default=0.0,
        help="T6/E3: PIGEON-style haversine label smoothing width in km for the cell "
             "classifier (0 = off, use one-hot masked CE). Try 50-200.",
    )
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument(
        "--loc-align-weight", type=float, default=0.0,
        help="T6/E4: weight of the GeoCLIP image<->location InfoNCE loss (0 = off, no "
             "location encoder built). Try 0.2-0.5.",
    )
    parser.add_argument("--loc-temperature", type=float, default=0.1)
    parser.add_argument(
        "--region-weight", type=float, default=0.0,
        help="T6/E8: weight of the country->region CE loss (0 = no region head). The "
             "region scheme is read from <cells-dir>/fold{f}_region_cells.json (build with "
             "geo_cells_kmeans.py --region-target 200). Try 0.3-0.6.",
    )
    parser.add_argument("--region-decode-weight", type=float, default=1.0,
                        help="weight of the region log-prob term in the retrieval decode.")
    parser.add_argument("--hard-neg-weight", type=float, default=0.0,
                        help="T6/E9: weight of the visually-similar-but-geographically-far "
                             "embedding repulsion (attacks France<->Germany rural confusion). "
                             "Try 0.3-1.0.")
    parser.add_argument("--hard-neg-far-km", type=float, default=400.0)
    parser.add_argument("--hard-neg-margin", type=float, default=0.25)
    parser.add_argument(
        "--loc-decode-weight", type=float, default=0.0,
        help="T6/E4: weight of the location-encoder similarity term in the eval decode "
             "(0 = image-image retrieval only).",
    )
    parser.add_argument(
        "--contrastive-country-blend", type=float, default=0.0,
        help="T6: blend same-country positives into the geo-contrastive loss "
             "(0 = cell-only, current; 1 = country-only). OSV-5M suggests ~0.3-0.5.",
    )
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
    parser.add_argument(
        "--eval-every", type=int, default=1,
        help="Run the (expensive) retrieval eval only every N epochs; epoch 0 and the last "
             "3 epochs are always evaluated. Screening runs use e.g. 3.",
    )
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument(
        "--fix-orientation", action="store_true",
        help="T6: rotate the ~8.9%% of images flagged upside down in "
             "geo_dataset/orientation_flags.json by 180 deg (train + val).",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Varies batch-sampler seed + torch init RNG for ensemble diversity.")
    parser.add_argument(
        "--distill-country-npz", default="",
        help="T6/R8: path to a teacher_country_fold{f}.npz (from src.cache_teacher_country) "
             "with `filenames` + `probs` (N,12). If set, adds a KL(teacher || student) term "
             "on the country head -- distils the 4-model ensemble's better-calibrated country "
             "posterior into the single compliant student (Further_Analysis §4.5).")
    parser.add_argument("--distill-weight", type=float, default=0.0,
                        help="weight of the country-distillation KL term. Try 0.5-2.0.")
    parser.add_argument(
        "--strong-aug", action="store_true",
        help="T6/R9 (Further_Analysis §4.2): wide RandomResizedCrop scale (0.3-1.0) + "
             "TrivialAugmentWide on the global view. For the small (11k) dataset / early "
             "convergence. No horizontal flip (keeps driving-side/text cues).")
    parser.add_argument("--exp-name", required=True)
    args = parser.parse_args()
    torch.manual_seed(1234 + args.seed)
    np.random.seed(1234 + args.seed)
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
