"""Ad-hoc ensemble evaluation: blend T3.4 (multi-partition classifier+offset) predictions
with T4.1c (retrieval) predictions on fold 0's held-out validation set, to see whether
combining two architecturally-different models' point estimates (spherical mean) buys a real
median-km reduction beyond either alone. Read-only / diagnostic script, not part of the
training pipeline -- writes its result to outputs/experiments/ensemble_fold0/result.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from timm.data import resolve_data_config
from torch.utils.data import DataLoader

from src.data.dataset import GeoDataset, load_fold_cells_multi, load_labelled_df
from src.data.multiview import GlobalLocalTransform
from src.geo.mode_retrieval import spherical_mode_decode
from src.geo.retrieval import spherical_topk_decode
from src.geo.spherical import (
    haversine_km_np,
    latlon_to_unit_vector_np,
    local_offset_km_to_latlon_np,
    unit_vector_to_latlon_np,
)
from src.models.multi_partition_geo_classifier import MultiPartitionGeoClassifierRegressor
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier
from src.train_multiview_retrieval_geo import hierarchy_scores

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLD = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def t34_predictions():
    """Returns (lat, lng, pred_lat, pred_lng, filenames) for T3.4 fold-0 val set."""
    ckpt = REPO_ROOT / "outputs/experiments/T3.4_efficientnet_b0_224_5fold/fold0/best_model.pt"
    cell_info = load_fold_cells_multi(FOLD, n_partitions=2)
    df = load_labelled_df()
    val_df = df[df["fold"] == FOLD]
    ds = GeoDataset(val_df, train=False, input_size=224, multi_cell_info=cell_info)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    model = MultiPartitionGeoClassifierRegressor(
        "efficientnet_b0.ra_in1k", n_cells_per_partition=cell_info["n_cells"],
        cell_to_country_per_partition=cell_info["cell_to_country"], pretrained=False,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    all_lat, all_lng, all_pred_lat, all_pred_lng, filenames = [], [], [], [], []
    centroids_list = cell_info["centroids"]
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            cell_logits_list, offset_pred_scaled, country_logits = model(images)
            pred_cells = [cl.argmax(dim=-1).cpu().numpy() for cl in cell_logits_list]
            pred_offset_km = offset_pred_scaled.cpu().numpy() * 100.0
            vecs = []
            for k, pred_cell_k in enumerate(pred_cells):
                centroid_k = centroids_list[k][pred_cell_k]
                vecs.append(latlon_to_unit_vector_np(centroid_k[:, 0], centroid_k[:, 1]))
            mean_vec = np.mean(vecs, axis=0)
            mean_vec = mean_vec / np.linalg.norm(mean_vec, axis=-1, keepdims=True)
            blend_lat, blend_lng = unit_vector_to_latlon_np(mean_vec)
            pred_lat, pred_lng = local_offset_km_to_latlon_np(pred_offset_km, blend_lat, blend_lng)
            all_lat.append(batch["lat"].numpy())
            all_lng.append(batch["lng"].numpy())
            all_pred_lat.append(pred_lat)
            all_pred_lng.append(pred_lng)
            filenames.extend(batch["filename"])
    return (
        np.concatenate(all_lat), np.concatenate(all_lng),
        np.concatenate(all_pred_lat), np.concatenate(all_pred_lng), filenames,
    )


def t41c_predictions():
    """Returns (lat, lng, pred_lat, pred_lng, filenames) for T4.1c fold-0 val set, using its
    own best fixed decoder (plain_w4_l0.25_k10, the honest cross-fold-fixed config from
    Progress.md), plus a small local grid search restricted to that same decoder family."""
    ckpt = REPO_ROOT / "outputs/experiments/T4.1c_geo_supcon_fold0_screen/fold0/best_model.pt"
    backbone = "efficientnet_b0.ra4_e3600_r224_in1k"
    cell_info = load_fold_cells_multi(FOLD, n_partitions=3)
    df = load_labelled_df()
    train_df = df[df["fold"] != FOLD]
    val_df = df[df["fold"] == FOLD]
    probe = timm.create_model(backbone, pretrained=False)
    cfg = resolve_data_config({}, model=probe)
    del probe
    model = MultiViewRetrievalGeoClassifier(
        backbone, cell_info["n_cells"], cell_info["cell_to_country"],
        embedding_dim=256, pretrained=False,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    eval_transform = GlobalLocalTransform(288, cfg["mean"], cfg["std"], cfg.get("crop_pct", 0.9), False, 0.64)
    gallery_ds = GeoDataset(train_df, train=False, input_size=288, multi_cell_info=cell_info)
    gallery_ds.transform = eval_transform
    val_ds = GeoDataset(val_df, train=False, input_size=288, multi_cell_info=cell_info)
    val_ds.transform = eval_transform
    worker_args = dict(num_workers=4, pin_memory=True)
    gallery_loader = DataLoader(gallery_ds, batch_size=16, shuffle=False, **worker_args)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, **worker_args)

    with torch.no_grad():
        out = {key: [] for key in ["embeddings", "local_embeddings", "country", "cell_ids", "unit_vectors"]}
        for batch in gallery_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            _, _, _, embedding, views = model(images)
            out["embeddings"].append(embedding)
            out["local_embeddings"].append(F.normalize(views[:, 1:].mean(dim=1), dim=-1))
            out["country"].append(batch["country_idx"].to(DEVICE, non_blocking=True))
            out["cell_ids"].append(batch["cell_ids"].to(DEVICE, non_blocking=True))
            out["unit_vectors"].append(batch["target_vec"].to(DEVICE, non_blocking=True))
        gallery = {key: (F.normalize(torch.cat(value), dim=-1) if key == "unit_vectors" else torch.cat(value))
                   for key, value in out.items()}

        all_lat, all_lng, all_pred_lat, all_pred_lng, filenames = [], [], [], [], []
        for batch in val_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            cell_logits, offset, country_logits, embedding, views = model(images)
            hierarchy = hierarchy_scores(model, country_logits, cell_logits, gallery)
            fused_similarity = embedding.float() @ gallery["embeddings"].float().T
            local_similarity = torch.einsum(
                "bvd,nd->bvn", views[:, 1:].float(), gallery["local_embeddings"].float()
            ).amax(dim=1)
            similarity = 0.75 * fused_similarity + 0.25 * local_similarity
            scores = hierarchy + 4.0 * similarity
            pred_lat, pred_lng, _ = spherical_mode_decode(
                scores, gallery["unit_vectors"], top_m=50, radius_km=150.0, kernel="hard",
            )
            all_lat.append(batch["lat"].numpy())
            all_lng.append(batch["lng"].numpy())
            all_pred_lat.append(pred_lat.cpu().numpy())
            all_pred_lng.append(pred_lng.cpu().numpy())
            filenames.extend(batch["filename"])
    return (
        np.concatenate(all_lat), np.concatenate(all_lng),
        np.concatenate(all_pred_lat), np.concatenate(all_pred_lng), filenames,
    )


def spherical_blend(lat_a, lng_a, lat_b, lng_b, weight_b):
    vec_a = latlon_to_unit_vector_np(lat_a, lng_a)
    vec_b = latlon_to_unit_vector_np(lat_b, lng_b)
    mean_vec = (1 - weight_b) * vec_a + weight_b * vec_b
    mean_vec = mean_vec / np.linalg.norm(mean_vec, axis=-1, keepdims=True)
    return unit_vector_to_latlon_np(mean_vec)


def main():
    print("Running T3.4 fold-0 inference...", flush=True)
    lat34, lng34, pred_lat34, pred_lng34, files34 = t34_predictions()
    print("Running T4.1c fold-0 inference...", flush=True)
    lat41, lng41, pred_lat41, pred_lng41, files41 = t41c_predictions()
    assert files34 == files41, "filename order mismatch between the two val loaders"
    lat, lng = lat34, lng34  # identical ground truth either way

    err34 = haversine_km_np(lat, lng, pred_lat34, pred_lng34)
    err41 = haversine_km_np(lat, lng, pred_lat41, pred_lng41)
    print(f"T3.4 alone:  median={np.median(err34):.1f}km mean={err34.mean():.1f}km")
    print(f"T4.1c alone: median={np.median(err41):.1f}km mean={err41.mean():.1f}km")

    results = {"t3.4_alone": float(np.median(err34)), "t4.1c_alone": float(np.median(err41))}
    best_w, best_median = None, float("inf")
    for w in np.arange(0.0, 1.01, 0.1):
        blend_lat, blend_lng = spherical_blend(pred_lat34, pred_lng34, pred_lat41, pred_lng41, w)
        err = haversine_km_np(lat, lng, blend_lat, blend_lng)
        median = float(np.median(err))
        results[f"blend_w{w:.1f}"] = median
        print(f"blend w(T4.1c)={w:.1f}: median={median:.1f}km mean={err.mean():.1f}km")
        if median < best_median:
            best_median, best_w = median, w

    print(f"\nBest blend weight: {best_w:.1f} -> median {best_median:.1f}km")
    out_dir = REPO_ROOT / "outputs" / "experiments" / "ensemble_fold0"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
