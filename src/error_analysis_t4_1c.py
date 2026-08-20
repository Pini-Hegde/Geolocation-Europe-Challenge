"""Per-country / per-image error breakdown for the T4.1c full 5-fold CV models.

Builds Phase 4's still-missing error-analysis tooling, applied to the current best model.
Uses ONE FIXED decoder config (plain_w4_l0.25_k10 -- the honest single-config CV winner,
151.06 +/- 3.74km, chosen because per-fold "best decoder" selection used that fold's own
validation labels and isn't available on the real holdout) across all 5 folds, so the
concatenated per-image errors give full out-of-fold coverage of every training image under
the config that will actually be used for submission -- not a cherry-picked number.

Usage:
    python -m src.error_analysis_t4_1c \
        --exp-name T4.1c_geo_supcon_5fold --backbone-exp T4.1b_multiview_mode_5fold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.dataset import load_fold_cells_multi
from src.geo.countries import COUNTRIES
from src.geo.retrieval import conditional_cell_log_probs, spherical_topk_decode
from src.geo.spherical import haversine_km_np
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier
from src.train_multiview_retrieval_geo import extract_gallery, hierarchy_scores, make_multiview_loaders

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "outputs" / "experiments"


@torch.no_grad()
def evaluate_fold(exp_name: str, fold: int, args) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, gallery_loader, val_loader, cell_info, _ = make_multiview_loaders(args, fold)
    model = MultiViewRetrievalGeoClassifier(
        args.backbone, cell_info["n_cells"], cell_info["cell_to_country"],
        embedding_dim=args.embedding_dim, pretrained=False,
    ).to(device)
    ckpt = OUT_DIR_BASE / exp_name / f"fold{fold}" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    gallery = extract_gallery(model, gallery_loader, device)

    rows = []
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        cell_logits, _, country_logits, embedding, views = model(images)
        hierarchy = hierarchy_scores(model, country_logits, cell_logits, gallery)
        fused_similarity = embedding.float() @ gallery["embeddings"].float().T
        local_similarity = torch.einsum(
            "bvd,nd->bvn", views[:, 1:].float(), gallery["local_embeddings"].float()
        ).amax(dim=1)
        similarity = (1 - args.local_weight) * fused_similarity + args.local_weight * local_similarity
        scores = hierarchy + args.retrieval_weight * similarity
        pred_lat, pred_lng, _ = spherical_topk_decode(scores, gallery["unit_vectors"], top_k=10)

        true_lat, true_lng = batch["lat"].numpy(), batch["lng"].numpy()
        errors_km = haversine_km_np(true_lat, true_lng, pred_lat.cpu().numpy(), pred_lng.cpu().numpy())
        pred_country = country_logits.argmax(dim=-1).cpu().numpy()
        true_country = batch["country_idx"].numpy()
        for i in range(len(errors_km)):
            rows.append({
                "filename": batch["filename"][i], "fold": fold,
                "true_country": COUNTRIES[true_country[i]],
                "pred_country": COUNTRIES[pred_country[i]],
                "error_km": float(errors_km[i]),
                "true_lat": float(true_lat[i]), "true_lng": float(true_lng[i]),
                "pred_lat": float(pred_lat[i].item()), "pred_lng": float(pred_lng[i].item()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="T4.1c_geo_supcon_5fold")
    parser.add_argument("--backbone", default="efficientnet_b0.ra4_e3600_r224_in1k")
    parser.add_argument("--n-partitions", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--local-fraction", type=float, default=0.64)
    parser.add_argument("--batch-size", type=int, default=32)  # unused for eval, kept for loader fn
    parser.add_argument("--samples-per-cell", type=int, default=4)  # unused for eval
    # fixed decoder config: honest single-config CV winner (plain_w4_l0.25_k10)
    parser.add_argument("--retrieval-weight", type=float, default=4.0)
    parser.add_argument("--local-weight", type=float, default=0.25)
    args = parser.parse_args()

    all_rows = []
    for fold in range(5):
        print(f"evaluating fold {fold}...", flush=True)
        df = evaluate_fold(args.exp_name, fold, args)
        all_rows.append(df)
        print(f"  fold {fold}: median {df['error_km'].median():.2f}km, n={len(df)}", flush=True)
    full = pd.concat(all_rows, ignore_index=True)

    out_dir = OUT_DIR_BASE / args.exp_name / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_dir / "per_image_errors.csv", index=False)

    per_country = full.groupby("true_country")["error_km"].agg(
        median_km="median", mean_km="mean", count="count",
        frac_lt_200km=lambda s: (s < 200).mean(), frac_lt_750km=lambda s: (s < 750).mean(),
    ).sort_values("median_km", ascending=False)
    per_country.to_csv(out_dir / "per_country_breakdown.csv")

    confusion = pd.crosstab(full["true_country"], full["pred_country"])
    confusion.to_csv(out_dir / "country_confusion_matrix.csv")

    print(f"\nOverall (fixed decoder retrieval_weight={args.retrieval_weight}, "
          f"local_weight={args.local_weight}): median={full['error_km'].median():.2f}km, "
          f"mean={full['error_km'].mean():.2f}km, n={len(full)}")
    print("\nPer-country breakdown (worst first):")
    print(per_country.to_string(float_format=lambda x: f"{x:.1f}"))
    print(f"\nWritten to {out_dir}/")


if __name__ == "__main__":
    main()
