"""
Honest CV evaluation for the T4.1b/c/T6 multi-view retrieval model, with:
  - one or more checkpoints fused in *score space* (CPlaNet Eq. 3 style) -> ensemble eval,
  - optional test-time augmentation (horizontal flip, multi-scale),
  - a full decoder-hyperparameter grid,
  - a per-country error breakdown.

This mirrors src/predict.py's decode so the decoder config picked here transfers to the
holdout submission. Run per fold; average the fold medians for the headline number.

Usage:
    python -m src.eval_cv --fold 0 --input-size 384 --fix-orientation \\
        --checkpoints outputs/experiments/T6_E2_stack_res384_fold0/fold0/best_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from src.geo.countries import COUNTRIES
from src.geo.mode_retrieval import spherical_mode_decode
from src.geo.retrieval import spherical_topk_decode
from src.geo.spherical import haversine_km_np, local_offset_km_to_latlon_np
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier
from src.train_multiview_retrieval_geo import (
    hierarchy_scores,
    make_multiview_loaders,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _apply_offset(lat, lng, off_km, weight):
    """Shift a decoded (lat, lng) by `weight` * the model's predicted within-cell offset
    (dx, dy) in km. weight=0 is a no-op. OSV-5M-style cell + relative-coordinate hybrid."""
    if weight == 0:
        return lat, lng
    return local_offset_km_to_latlon_np(off_km * weight, lat, lng)


@torch.no_grad()
def _forward(model, images, tta_flip: bool, tta_scales: list[float]):
    """Return (cell_logits[list], country_logits, fused_embedding, view_embeddings),
    averaged over the requested TTA transforms. Logits are averaged in probability space."""
    variants = [images]
    if tta_flip:
        variants.append(torch.flip(images, dims=[-1]))
    scaled = []
    for s in tta_scales:
        if abs(s - 1.0) < 1e-6:
            continue
        size = int(round(images.shape[-1] * s / 32) * 32)
        for v in list(variants):
            scaled.append(
                F.interpolate(
                    v.flatten(0, 1), size=size, mode="bicubic", align_corners=False
                ).reshape(*v.shape[:2], v.shape[2], size, size)
            )
    variants += scaled

    country_p, cell_p, emb, views, off = None, None, None, None, None
    for v in variants:
        cl, o, ctry, e, vw = model(v)
        cp = F.softmax(ctry.float(), dim=-1)
        clp = [F.softmax(x.float(), dim=-1) for x in cl]
        country_p = cp if country_p is None else country_p + cp
        cell_p = clp if cell_p is None else [a + b for a, b in zip(cell_p, clp)]
        emb = e if emb is None else emb + e
        views = vw if views is None else views + vw
        off = o if off is None else off + o
    n = len(variants)
    country_logits = (country_p / n).clamp_min(1e-9).log()
    cell_logits = [(p / n).clamp_min(1e-9).log() for p in cell_p]
    emb = F.normalize(emb / n, dim=-1)
    views = F.normalize(views / n, dim=-1)
    return cell_logits, country_logits, emb, views, off / n


@torch.no_grad()
def extract_gallery_tta(model, loader, device, tta_flip, tta_scales):
    model.eval()
    has_loc = getattr(model, "use_location_encoder", False)
    keys = ["embeddings", "local_embeddings", "country", "cell_ids", "unit_vectors"]
    if has_loc:
        keys.append("loc_embeddings")
    out = {k: [] for k in keys}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        _, _, emb, views, _ = _forward(model, images, tta_flip, tta_scales)
        out["embeddings"].append(emb)
        out["local_embeddings"].append(F.normalize(views[:, 1:].mean(dim=1), dim=-1))
        out["country"].append(batch["country_idx"].to(device))
        out["cell_ids"].append(batch["cell_ids"].to(device))
        out["unit_vectors"].append(batch["target_vec"].to(device))
        if has_loc:
            out["loc_embeddings"].append(
                model.encode_location(batch["lat"].float().to(device),
                                      batch["lng"].float().to(device)))
    return {
        k: F.normalize(torch.cat(v), dim=-1) if k == "unit_vectors" else torch.cat(v)
        for k, v in out.items()
    }


def _load_models(paths, backbone, cells, embedding_dim, dropout, device, cell_proj_dim=0,
                 use_location_encoder=False, country_head_hidden=0):
    models = []
    for p in paths:
        sd = torch.load(p, map_location=device)
        has_loc = use_location_encoder or any(k.startswith("location_encoder.") for k in sd)
        chh = country_head_hidden or (
            sd["country_head.0.weight"].shape[0] if "country_head.0.weight" in sd else 0
        )
        m = MultiViewRetrievalGeoClassifier(
            backbone, cells["n_cells"], cells["cell_to_country"],
            embedding_dim=embedding_dim, pretrained=False, dropout=dropout,
            cell_proj_dim=(cell_proj_dim or None),
            use_location_encoder=has_loc, country_head_hidden=chh,
        ).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--backbone", default="efficientnet_b0.ra4_e3600_r224_in1k")
    ap.add_argument("--n-partitions", type=int, default=4)
    ap.add_argument("--cells-dir", default="", help="e.g. outputs/protocol/geo_cells_multi_mid")
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--cell-proj-dim", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--local-fraction", type=float, default=0.64)
    ap.add_argument("--eval-batch-size", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--fix-orientation", action="store_true")
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--tta-scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--retrieval-weights", type=float, nargs="+", default=[2, 4, 6, 8, 12])
    ap.add_argument("--local-weights", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--radii-km", type=float, nargs="+", default=[100, 150, 200, 250])
    ap.add_argument("--top-ms", type=int, nargs="+", default=[50])
    ap.add_argument("--kernels", nargs="+", default=["hard", "gaussian"])
    ap.add_argument("--top-ks", type=int, nargs="+", default=[10])
    ap.add_argument("--geomed-iters", type=int, nargs="+", default=[0],
                    help="Weiszfeld geometric-median iterations for the point estimate "
                         "(0 = spherical mean; the geometric median is the exact minimiser "
                         "of expected haversine and robust to far outliers). Try 0 8.")
    ap.add_argument("--offset-weights", type=float, nargs="+", default=[0.0],
                    help="OSV-5M cell+relative hybrid: fraction of the predicted within-cell "
                         "offset to add to the decoded point. Try 0 0.3 0.6 1.0.")
    ap.add_argument("--country-boosts", type=float, nargs="+", default=[0.0],
                    help="Extra multiplier on log P(country) added to the gallery scores -- "
                         "counters retrieval crossing borders when the country head is weak "
                         "(France 0.58). Try 0 3 6 10 20.")
    ap.add_argument("--country-topk", type=int, nargs="+", default=[0],
                    help="Hard-mask the retrieval gallery to the query's top-k predicted "
                         "countries (0 = off). Try 0 1 2 3.")
    ap.add_argument("--loc-weight", type=float, default=0.0,
                    help="E4: weight of the location-encoder similarity term in the retrieval "
                         "score (0 = image-image only; needs a checkpoint with a loc encoder).")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_args = SimpleNamespace(
        backbone=args.backbone, folds=[args.fold], n_partitions=args.n_partitions,
        cells_dir=args.cells_dir, input_size=args.input_size,
        local_fraction=args.local_fraction,
        batch_size=args.eval_batch_size, samples_per_cell=4,
        eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
        embedding_dim=args.embedding_dim, dropout=args.dropout,
        fix_orientation=args.fix_orientation, label_smooth_tau=0.0,
    )
    _, gallery_loader, val_loader, cells, _ = make_multiview_loaders(loader_args, args.fold)
    models = _load_models(
        args.checkpoints, args.backbone, cells, args.embedding_dim, args.dropout, device,
        args.cell_proj_dim,
    )
    print(f"{len(models)} model(s); fold {args.fold}; tta_flip={args.tta_flip} "
          f"scales={args.tta_scales}", flush=True)

    galleries = [
        extract_gallery_tta(m, gallery_loader, device, args.tta_flip, args.tta_scales)
        for m in models
    ]

    tags = [f"cb{cb:g}_ck{ck}" for cb in args.country_boosts for ck in args.country_topk]
    err = {
        f"mode_w{w:g}_l{lw:g}_r{r:g}_{k}_m{m}_o{o:g}_gm{gm}_{t}": []
        for w in args.retrieval_weights for lw in args.local_weights
        for r in args.radii_km for k in args.kernels for m in args.top_ms
        for gm in args.geomed_iters
        for o in args.offset_weights for t in tags
    }
    err.update({
        f"plain_w{w:g}_l{lw:g}_k{kk}_o{o:g}_{t}": []
        for w in args.retrieval_weights for lw in args.local_weights
        for kk in args.top_ks for o in args.offset_weights for t in tags
    })
    countries_all = []

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        true_lat, true_lng = batch["lat"].numpy(), batch["lng"].numpy()
        countries_all.append(batch["country_idx"].numpy())
        # Fuse per-model hierarchy + retrieval similarities in score space.
        fused_hier = None
        fused_sim = None
        fused_local = None
        fused_off = None
        fused_clp = None  # log P(country) per query, averaged over models
        for mdl, gal in zip(models, galleries):
            cl, ctry, emb, views, off = _forward(mdl, images, args.tta_flip, args.tta_scales)
            h = hierarchy_scores(mdl, ctry, cl, gal)
            s = emb.float() @ gal["embeddings"].float().T
            if args.loc_weight > 0 and "loc_embeddings" in gal:
                s = s + args.loc_weight * (emb.float() @ gal["loc_embeddings"].float().T)
            ls = torch.einsum(
                "bvd,nd->bvn", views[:, 1:].float(), gal["local_embeddings"].float()
            ).amax(dim=1)
            clp = F.log_softmax(ctry.float(), dim=-1)
            fused_hier = h if fused_hier is None else fused_hier + h
            fused_sim = s if fused_sim is None else fused_sim + s
            fused_local = ls if fused_local is None else fused_local + ls
            fused_off = off if fused_off is None else fused_off + off
            fused_clp = clp if fused_clp is None else fused_clp + clp
        fused_hier /= len(models)
        fused_sim /= len(models)
        fused_local /= len(models)
        fused_clp /= len(models)
        fused_off = (fused_off / len(models)).float().cpu().numpy() * 100.0  # km (dx, dy)
        uv = galleries[0]["unit_vectors"]
        gal_country = galleries[0]["country"]  # (N,) country idx per gallery point
        clp_by_gallery = fused_clp[:, gal_country]  # (B, N) log P(country_n | image_b)
        # country-topk gate: which countries are in each query's top-k predictions
        topk_max = max(args.country_topk) if any(args.country_topk) else 0
        if topk_max:
            topk_countries = fused_clp.topk(topk_max, dim=-1).indices  # (B, topk_max)

        for w in args.retrieval_weights:
            for lw in args.local_weights:
                sim = (1 - lw) * fused_sim + lw * fused_local
                base = fused_hier + w * sim
                for cb in args.country_boosts:
                    for ck in args.country_topk:
                        scores = base + cb * clp_by_gallery
                        if ck > 0:
                            in_topk = (
                                gal_country[None, :] == topk_countries[:, :ck, None]
                            ).any(dim=1)  # (B, N)
                            scores = scores.masked_fill(~in_topk, -1e4)
                        tag = f"cb{cb:g}_ck{ck}"
                        for kk in args.top_ks:
                            plat, plng, _ = spherical_topk_decode(scores, uv, top_k=kk)
                            for ow in args.offset_weights:
                                alat, alng = _apply_offset(
                                    plat.cpu().numpy(), plng.cpu().numpy(), fused_off, ow)
                                err[f"plain_w{w:g}_l{lw:g}_k{kk}_o{ow:g}_{tag}"].extend(
                                    haversine_km_np(true_lat, true_lng, alat, alng))
                        for r in args.radii_km:
                          for k in args.kernels:
                            for m in args.top_ms:
                              for gm in args.geomed_iters:
                                    plat, plng, _ = spherical_mode_decode(
                                        scores, uv, top_m=m, radius_km=r, kernel=k,
                                        geometric_median_iters=gm)
                                    pn, gn = plat.cpu().numpy(), plng.cpu().numpy()
                                    for ow in args.offset_weights:
                                        alat, alng = _apply_offset(pn, gn, fused_off, ow)
                                        err[
                                            f"mode_w{w:g}_l{lw:g}_r{r:g}_{k}_m{m}_o{ow:g}_gm{gm}_{tag}"
                                        ].extend(haversine_km_np(true_lat, true_lng, alat, alng))

    countries_all = np.concatenate(countries_all)
    stats = {}
    for key, vals in err.items():
        vals = np.asarray(vals)
        stats[key] = {
            "median_km": float(np.median(vals)), "mean_km": float(vals.mean()),
            "frac_lt_200km": float(np.mean(vals < 200)),
            "frac_lt_750km": float(np.mean(vals < 750)),
        }
    best = min(stats, key=lambda k: stats[k]["median_km"])
    print(f"\nBEST: {best}  median={stats[best]['median_km']:.1f}  "
          f"mean={stats[best]['mean_km']:.0f}  <200={stats[best]['frac_lt_200km']:.3f}")
    print("\nTop 15 decoders:")
    for k, v in sorted(stats.items(), key=lambda x: x[1]["median_km"])[:15]:
        print(f"  {k:44s} med={v['median_km']:6.1f} mean={v['mean_km']:5.0f}")

    best_err = np.asarray(err[best])
    per_country = {}
    print("\nPer-country (best decoder):")
    for ci, cname in enumerate(COUNTRIES):
        mask = countries_all == ci
        if mask.sum():
            per_country[cname] = {
                "n": int(mask.sum()), "median_km": float(np.median(best_err[mask])),
                "mean_km": float(best_err[mask].mean()),
            }
            print(f"  {cname:16s} n={mask.sum():4d}  median={per_country[cname]['median_km']:6.1f}"
                  f"  mean={per_country[cname]['mean_km']:6.0f}")

    payload = {
        "checkpoints": args.checkpoints, "fold": args.fold,
        "tta_flip": args.tta_flip, "tta_scales": args.tta_scales,
        "fix_orientation": args.fix_orientation,
        "best_decoder": best, "best": stats[best], "per_country": per_country,
        "all": stats,
    }
    out = args.output or (
        REPO_ROOT / "outputs" / "experiments" / f"eval_cv_fold{args.fold}.json"
    )
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
