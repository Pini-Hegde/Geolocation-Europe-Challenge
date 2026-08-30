"""
Score-space ensemble of heterogeneous multi-view retrieval checkpoints (CPlaNet Eq. 3 style),
for fold-0 CV. Each checkpoint may use a different cell scheme / K / embedding dim / extra
heads -- they are fused at the level of the per-gallery-point score vector, which is the same
shape (n_val, n_gallery) for every model because the training-image gallery and the
validation set are the identical sets, in the identical order, for a given fold.

Per model the score for gallery point n given query b is
    log P(country_n) + mean_k log P(cell_{k,n} | country_n) + w * (emb_b . emb_n)
(+ w_loc * (emb_b . loc_emb_n) when the model has a location encoder). The ensemble sums
these across models, then mode-decodes.

Usage:
    python -m src.ensemble_cv --fold 0 \\
        --spec outputs/experiments/T6_E3e_cells368_hsmooth45_fold0/fold0/best_model.pt:geo_cells_multi_fine:4:224:224 \\
        --spec outputs/experiments/T6_E3d_cells192_hsmooth65_fold0/fold0/best_model.pt:geo_cells_multi_mid:3:0:160 \\
        --retrieval-weights 8 16 --radii-km 150 200 250 300
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
from src.geo.spherical import haversine_km_np
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier
from src.train_multiview_retrieval_geo import hierarchy_scores, make_multiview_loaders

REPO_ROOT = Path(__file__).resolve().parents[1]


def _infer_arch(sd):
    n_part = len([k for k in sd if k.startswith("cls_heads.") and k.endswith(".weight")])
    emb = sd["embedding_head.weight"].shape[0]
    proj = sd["cell_trunk.0.weight"].shape[0] if "cell_trunk.0.weight" in sd else 0
    chh = sd["country_head.0.weight"].shape[0] if "country_head.0.weight" in sd else 0
    loc = any(k.startswith("location_encoder.") for k in sd)
    return n_part, emb, proj, chh, loc


@torch.no_grad()
def model_components(spec, fold, device, wloc, tta_flip):
    """Extract per-model score components ONCE. Returns a dict with:
       hier (n_val, n_gal), imgsim (n_val, n_gal), locsim or None,
       tlat, tlng, tctry (n_val,), gallery unit vectors (n_gal, 3)."""
    sd = torch.load(spec["ckpt"], map_location=device)
    n_part, emb, proj, chh, loc = _infer_arch(sd)
    la = SimpleNamespace(
        backbone="efficientnet_b0.ra4_e3600_r224_in1k", folds=[fold], n_partitions=n_part,
        cells_dir=str(REPO_ROOT / "outputs" / "protocol" / spec["cells"]),
        input_size=288, local_fraction=0.64, batch_size=24, samples_per_cell=4,
        eval_batch_size=24, num_workers=8, embedding_dim=emb, dropout=0.3,
        fix_orientation=True, label_smooth_tau=0.0)
    _, gl, vl, cells, _ = make_multiview_loaders(la, fold)
    m = MultiViewRetrievalGeoClassifier(
        la.backbone, cells["n_cells"], cells["cell_to_country"], embedding_dim=emb,
        pretrained=False, dropout=0.3, cell_proj_dim=(proj or None),
        use_location_encoder=loc, country_head_hidden=chh).to(device)
    m.load_state_dict(sd)
    m.eval()

    def fwd(img):
        cl, off, ctry, e, vw = m(img)
        if tta_flip:
            cl2, _, ctry2, e2, _ = m(torch.flip(img, dims=[-1]))
            cl = [(a.softmax(-1) + b.softmax(-1)).clamp_min(1e-9).log() for a, b in zip(cl, cl2)]
            ctry = (ctry.softmax(-1) + ctry2.softmax(-1)).clamp_min(1e-9).log()
            e = F.normalize(e + e2, dim=-1)
        return cl, ctry, e

    g = {k: [] for k in ["emb", "loc", "country", "cell_ids", "uv"]}
    for b in gl:
        cl, ctry, e = fwd(b["image"].to(device))
        g["emb"].append(e)
        g["country"].append(b["country_idx"].to(device))
        g["cell_ids"].append(b["cell_ids"].to(device))
        g["uv"].append(b["target_vec"].to(device))
        if loc:
            g["loc"].append(m.encode_location(b["lat"].float().to(device), b["lng"].float().to(device)))
    gal = {"embeddings": torch.cat(g["emb"]), "country": torch.cat(g["country"]),
           "cell_ids": torch.cat(g["cell_ids"]),
           "unit_vectors": F.normalize(torch.cat(g["uv"]), dim=-1)}
    gal_loc = torch.cat(g["loc"]) if loc else None

    hier, imgsim, locsim, tlat, tlng, tctry, topconf = [], [], [], [], [], [], []
    for b in vl:
        cl, ctry, e = fwd(b["image"].to(device))
        hier.append(hierarchy_scores(m, ctry, cl, gal).cpu())
        imgsim.append((e.float() @ gal["embeddings"].float().T).cpu())
        topconf.append(ctry.softmax(-1).max(-1).values.cpu())
        if loc and gal_loc is not None:
            locsim.append((e.float() @ gal_loc.float().T).cpu())
        tlat.append(b["lat"].numpy()); tlng.append(b["lng"].numpy())
        tctry.append(b["country_idx"].numpy())
    return {
        "hier": torch.cat(hier), "imgsim": torch.cat(imgsim),
        "locsim": torch.cat(locsim) if locsim else None,
        "topconf": torch.cat(topconf),
        "tlat": np.concatenate(tlat), "tlng": np.concatenate(tlng),
        "tctry": np.concatenate(tctry), "uv": gal["unit_vectors"].cpu(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--spec", action="append", required=True,
                    help="ckpt:cells_subdir:_:_:_  (arch auto-inferred; last 3 fields ignored, "
                         "kept for readability)")
    ap.add_argument("--retrieval-weights", type=float, nargs="+", default=[8, 16])
    ap.add_argument("--loc-weight", type=float, default=0.5)
    ap.add_argument("--radii-km", type=float, nargs="+", default=[150, 200, 250, 300])
    ap.add_argument("--top-ms", type=int, nargs="+", default=[100])
    ap.add_argument("--kernels", nargs="+", default=["hard"])
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--hier-weights", type=float, nargs="+", default=[1.0],
                    help="Multiplier on the country/cell hierarchy log-prob term. <1 leans on "
                         "pure visual retrieval -- helps countries the classifier gets wrong "
                         "(France 0.58) where visually-similar gallery scenes still exist. Try "
                         "1.0 0.5 0.25 0.0.")
    ap.add_argument("--conf-gate", type=float, nargs="+", default=[0.0],
                    help="If >0: for queries whose top country prob < this threshold, use "
                         "hier_weight=0 (pure retrieval); else use the swept hier_weight. "
                         "0 = off (apply hier_weight to all).")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="per-model ensemble weights (default equal)")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "experiments" / "ensemble_cv.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = []
    for s in args.spec:
        parts = s.split(":")
        specs.append({"ckpt": parts[0], "cells": parts[1]})
    mw = args.weights or [1.0] * len(specs)

    comps = []
    for i, sp in enumerate(specs):
        comps.append(model_components(sp, args.fold, device, args.loc_weight, args.tta_flip))
        print(f"model {i}: {sp['ckpt'].split('/')[-3]}  has_loc={comps[-1]['locsim'] is not None}",
              flush=True)
    tlat, tlng, tctry = comps[0]["tlat"], comps[0]["tlng"], comps[0]["tctry"]
    uv = comps[0]["uv"].to(device)
    # mean top-country confidence across models, per query (n_val,)
    conf = torch.stack([c["topconf"] for c in comps]).mean(0)

    stats, errs_by_key = {}, {}
    for wret in args.retrieval_weights:
        fused_hier = None
        fused_retr = None
        for i, c in enumerate(comps):
            retr = wret * c["imgsim"]
            if c["locsim"] is not None and args.loc_weight > 0:
                retr = retr + wret * args.loc_weight * c["locsim"]
            fused_hier = mw[i] * c["hier"] if fused_hier is None else fused_hier + mw[i] * c["hier"]
            fused_retr = mw[i] * retr if fused_retr is None else fused_retr + mw[i] * retr
        fused_hier = (fused_hier / sum(mw))
        fused_retr = (fused_retr / sum(mw))
        for hw in args.hier_weights:
            for cg in args.conf_gate:
                if cg > 0:
                    hwv = torch.where(conf < cg, torch.zeros_like(conf), torch.full_like(conf, hw))
                    scores = hwv[:, None] * fused_hier + fused_retr
                else:
                    scores = hw * fused_hier + fused_retr
                scores = scores.to(device)
                for r in args.radii_km:
                    for k in args.kernels:
                        for tm in args.top_ms:
                            plat, plng, _ = spherical_mode_decode(scores, uv, top_m=tm, radius_km=r, kernel=k)
                            e = haversine_km_np(tlat, tlng, plat.cpu().numpy(), plng.cpu().numpy())
                            key = f"w{wret:g}_hw{hw:g}_cg{cg:g}_r{r:g}_{k}_m{tm}"
                            errs_by_key[key] = e
                            stats[key] = {"median_km": float(np.median(e)), "mean_km": float(e.mean()),
                                          "frac_lt_200km": float(np.mean(e < 200))}
    best = min(stats, key=lambda x: stats[x]["median_km"])
    print(f"\nENSEMBLE ({len(specs)} models) BEST: {best}  median={stats[best]['median_km']:.1f}  "
          f"mean={stats[best]['mean_km']:.0f}")
    for k, v in sorted(stats.items(), key=lambda x: x[1]["median_km"])[:10]:
        print(f"  {k:28s} med={v['median_km']:6.1f} mean={v['mean_km']:5.0f}")
    be = errs_by_key[best]
    print("\nPer-country (ensemble best):")
    pc = {}
    for ci, cn in enumerate(COUNTRIES):
        mm = tctry == ci
        if mm.sum():
            pc[cn] = {"n": int(mm.sum()), "median_km": float(np.median(be[mm])), "mean_km": float(be[mm].mean())}
            print(f"  {cn:16s} n={mm.sum():4d} median={pc[cn]['median_km']:6.1f} mean={pc[cn]['mean_km']:6.0f}")
    args.output.write_text(json.dumps(
        {"specs": args.spec, "best": best, "stats": stats, "per_country": pc}, indent=2))
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
