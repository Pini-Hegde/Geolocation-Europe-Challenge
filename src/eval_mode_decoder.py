"""Evaluate zero-parameter geographic mode decoding on a trained retrieval checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.geo.mode_retrieval import spherical_mode_decode
from src.geo.retrieval import conditional_cell_log_probs, spherical_topk_decode
from src.geo.spherical import haversine_km_np
from src.models.retrieval_geo_classifier import RetrievalGeoClassifier
from src.train_retrieval_geo import extract_gallery, make_loaders


def hierarchy_scores(model, country_logits, cell_logits, gallery):
    scores = F.log_softmax(country_logits.float(), dim=-1)[:, gallery["country"]]
    for k, logits in enumerate(cell_logits):
        conditional = conditional_cell_log_probs(logits.float(), model.country_cell_mask(k))
        scores = scores + conditional[:, gallery["cell_ids"][:, k]] / model.n_partitions
    return scores


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--backbone", default="efficientnet_b0.ra4_e3600_r224_in1k")
    parser.add_argument("--n-partitions", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--retrieval-weights", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    parser.add_argument("--top-ms", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--radii-km", type=float, nargs="+", default=[25, 50, 100, 150, 250])
    parser.add_argument("--kernels", nargs="+", default=["gaussian", "hard"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, gallery_loader, val_loader, cell_info, preprocess = make_loaders(
        args.fold, args.n_partitions, args.backbone, args.input_size,
        args.batch_size, args.num_workers,
    )
    model = RetrievalGeoClassifier(
        args.backbone, cell_info["n_cells"], cell_info["cell_to_country"],
        embedding_dim=args.embedding_dim, pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    gallery = extract_gallery(model, gallery_loader, device)

    specs = [
        (weight, top_m, radius, kernel)
        for weight in args.retrieval_weights
        for top_m in args.top_ms
        for radius in args.radii_km
        for kernel in args.kernels
    ]
    errors = {f"w{w:g}_m{m}_r{r:g}_{kernel}": [] for w, m, r, kernel in specs}
    errors["plain_w4_k10"] = []

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        cell_logits, _, country_logits, embeddings = model(images)
        hierarchy = hierarchy_scores(model, country_logits, cell_logits, gallery)
        similarity = embeddings.float() @ gallery["embeddings"].float().T
        true_lat, true_lng = batch["lat"].numpy(), batch["lng"].numpy()

        plain_scores = hierarchy + 4.0 * similarity
        plain_lat, plain_lng, _ = spherical_topk_decode(
            plain_scores, gallery["unit_vectors"], top_k=10
        )
        errors["plain_w4_k10"].extend(haversine_km_np(
            true_lat, true_lng, plain_lat.cpu().numpy(), plain_lng.cpu().numpy()
        ).tolist())

        score_cache = {weight: hierarchy + weight * similarity for weight in args.retrieval_weights}
        for weight, top_m, radius, kernel in specs:
            pred_lat, pred_lng, _ = spherical_mode_decode(
                score_cache[weight], gallery["unit_vectors"], top_m=top_m,
                radius_km=radius, kernel=kernel,
            )
            key = f"w{weight:g}_m{top_m}_r{radius:g}_{kernel}"
            errors[key].extend(haversine_km_np(
                true_lat, true_lng, pred_lat.cpu().numpy(), pred_lng.cpu().numpy()
            ).tolist())

    results = {}
    for key, values in errors.items():
        values = np.asarray(values)
        results[key] = {
            "median_km": float(np.median(values)),
            "mean_km": float(np.mean(values)),
            "frac_lt_200km": float(np.mean(values < 200)),
            "frac_lt_750km": float(np.mean(values < 750)),
        }
    best = min(results, key=lambda key: results[key]["median_km"])
    payload = {
        "checkpoint": str(args.checkpoint), "fold": args.fold,
        "preprocess": preprocess, "best_decoder": best,
        "best": results[best], "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"best_decoder": best, **results[best]}, indent=2))
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
