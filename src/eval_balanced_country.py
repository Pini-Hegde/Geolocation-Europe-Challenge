"""
T6/R5 -- transductive balanced country assignment at decode (Further_Analysis_&_Findings §4.4a).

The fold-0 diagnosis (T6_FINDINGS, diag_e3e) is that France/Italy/Spain country accuracy
(~0.58-0.65) is the wall: ~45% of FR/ES images land >500 km because the country head is
*confidently* wrong. Per-image country-topk masking (eval_cv --country-topk) did nothing --
when France is wrong it isn't even in the model's top-2.

This is different: the val / holdout set is (near-)balanced by construction (fold-0 per-country
counts 173-200; the public holdout is almost certainly 200/country). So instead of a per-image
argmax we solve a GLOBAL assignment of queries to countries that respects that known marginal,
via Sinkhorn on the country log-probs. A confidently-wrong French image can still be pulled
back to France if it is among the least-confidently-German queries and Germany's quota is full.

Runs inference-only on any trained checkpoint. Reports baseline vs balanced overall + per
-country median, and country-assignment accuracy, so we can see whether the mechanism helps
before wiring it into predict.py.

Usage:
    python -m src.eval_balanced_country \
        --checkpoint outputs/experiments/T6_R1_mnv4_fullrank_fold0/fold0/best_model.pt \
        --backbone mobilenetv4_conv_small.e2400_r224_in1k \
        --cells-dir outputs/protocol/geo_cells_multi_fine --n-partitions 4 \
        --embedding-dim 256 --input-size 288 --fix-orientation \
        --retrieval-weight 12 --radius-km 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from src.geo.countries import COUNTRIES, N_COUNTRIES
from src.geo.mode_retrieval import spherical_mode_decode
from src.geo.spherical import haversine_km_np
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier
from src.train_multiview_retrieval_geo import (
    extract_gallery,
    hierarchy_scores,
    make_multiview_loaders,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def gated_sinkhorn(log_p: torch.Tensor, col_marginal: torch.Tensor, conf_thresh: float,
                   temperature: float = 1.0, iters: int = 200) -> torch.Tensor:
    """Confidence-gated balanced assignment. Queries whose max softmax prob >= `conf_thresh`
    are FROZEN at their argmax (protects the accurate countries -- UK/Iceland/Finland -- that
    a symmetric Sinkhorn robs to fill other quotas). Sinkhorn runs only over the uncertain
    remainder against the residual marginal (target minus frozen counts)."""
    p = log_p.exp()
    conf, argm = p.max(1)
    frozen = conf >= conf_thresh
    Q = torch.zeros_like(log_p)
    Q[frozen, argm[frozen]] = 1.0
    if (~frozen).any():
        frozen_counts = Q[frozen].sum(0) if frozen.any() else torch.zeros_like(col_marginal)
        residual = (col_marginal - frozen_counts).clamp_min(0.0)
        Qr = sinkhorn_assignment(log_p[~frozen], residual, temperature=temperature, iters=iters)
        Q[~frozen] = Qr
    return Q


def sinkhorn_assignment(log_p: torch.Tensor, col_marginal: torch.Tensor,
                        temperature: float = 1.0, iters: int = 200) -> torch.Tensor:
    """Balanced transport of `N` queries to `C` countries.

    `log_p`: (N, C) query country log-probabilities (need not be calibrated).
    `col_marginal`: (C,) desired number of queries per country (sums to N).
    Returns Q: (N, C) doubly-constrained coupling -- each row sums to 1, each column sums to
    `col_marginal[c]`. Hard assignment = Q.argmax(1).
    """
    N = log_p.shape[0]
    logK = log_p / max(temperature, 1e-6)
    log_r = torch.zeros(N, device=log_p.device)  # row quota = 1 each
    log_c = torch.zeros(N_COUNTRIES, device=log_p.device)
    log_row_target = torch.zeros(N, device=log_p.device)
    log_col_target = torch.log(col_marginal.clamp_min(1e-9))
    for _ in range(iters):
        # row normalise to 1
        log_r = log_row_target - torch.logsumexp(logK + log_c[None, :], dim=1)
        # col normalise to col_marginal
        log_c = log_col_target - torch.logsumexp(logK + log_r[:, None], dim=0)
    return torch.exp(logK + log_r[:, None] + log_c[None, :])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--backbone", default="mobilenetv4_conv_small.e2400_r224_in1k")
    ap.add_argument("--n-partitions", type=int, default=4)
    ap.add_argument("--cells-dir", default="outputs/protocol/geo_cells_multi_fine")
    ap.add_argument("--embedding-dim", type=int, default=256)
    ap.add_argument("--cell-proj-dim", type=int, default=0)
    ap.add_argument("--input-size", type=int, default=288)
    ap.add_argument("--local-fraction", type=float, default=0.64)
    ap.add_argument("--eval-batch-size", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--fix-orientation", action="store_true")
    ap.add_argument("--retrieval-weight", type=float, default=12.0)
    ap.add_argument("--radius-km", type=float, default=150.0)
    ap.add_argument("--top-m", type=int, default=50)
    ap.add_argument("--kernel", default="hard")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--conf-threshs", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.4],
                    help="freeze queries with max country prob >= thresh at their argmax; "
                         "1.0 = ungated (symmetric Sinkhorn).")
    ap.add_argument("--marginal", choices=["true", "uniform"], default="true",
                    help="true = the actual val per-country counts (optimistic, validates the "
                         "mechanism); uniform = N/12 each (what we'd assume on the 200/country "
                         "holdout).")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "outputs" / "experiments" / "eval_balanced_country.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_args = SimpleNamespace(
        backbone=args.backbone, folds=[args.fold], n_partitions=args.n_partitions,
        cells_dir=args.cells_dir, input_size=args.input_size,
        local_fraction=args.local_fraction, batch_size=args.eval_batch_size,
        samples_per_cell=4, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, embedding_dim=args.embedding_dim,
        dropout=0.3, fix_orientation=args.fix_orientation, label_smooth_tau=0.0,
        seed=0,
    )
    _, gallery_loader, val_loader, cells, _ = make_multiview_loaders(loader_args, args.fold)
    sd = torch.load(args.checkpoint, map_location=device)
    model = MultiViewRetrievalGeoClassifier(
        args.backbone, cells["n_cells"], cells["cell_to_country"],
        embedding_dim=args.embedding_dim, pretrained=False, dropout=0.3,
        cell_proj_dim=(args.cell_proj_dim or None),
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    gallery = extract_gallery(model, gallery_loader, device)
    gal_country = gallery["country"]  # (Ngal,)

    all_scores, all_clp, all_true_lat, all_true_lng, all_true_country = [], [], [], [], []
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        cell_logits, offset, country_logits, embedding, views = model(images)
        h = hierarchy_scores(model, country_logits, cell_logits, gallery)
        s = h + args.retrieval_weight * (embedding.float() @ gallery["embeddings"].float().T)
        all_scores.append(s.cpu())
        all_clp.append(F.log_softmax(country_logits.float(), dim=-1).cpu())
        all_true_lat.append(batch["lat"].numpy())
        all_true_lng.append(batch["lng"].numpy())
        all_true_country.append(batch["country_idx"].numpy())

    scores = torch.cat(all_scores).to(device)          # (Nval, Ngal)
    clp = torch.cat(all_clp).to(device)                # (Nval, 12)
    true_lat = np.concatenate(all_true_lat)
    true_lng = np.concatenate(all_true_lng)
    true_country = np.concatenate(all_true_country)
    n_val = len(true_country)

    def decode(sc):
        plat, plng, _ = spherical_mode_decode(
            sc, gallery["unit_vectors"], top_m=args.top_m,
            radius_km=args.radius_km, kernel=args.kernel)
        return haversine_km_np(true_lat, true_lng, plat.cpu().numpy(), plng.cpu().numpy())

    def summarise(errs, assigned_country):
        rep = {"median_km": float(np.median(errs)), "mean_km": float(errs.mean()),
               "frac_lt_200km": float(np.mean(errs < 200)),
               "country_acc": float(np.mean(assigned_country == true_country)),
               "per_country": {}}
        for ci, cn in enumerate(COUNTRIES):
            m = true_country == ci
            if m.sum():
                rep["per_country"][cn] = {
                    "n": int(m.sum()), "median_km": round(float(np.median(errs[m])), 1),
                    "country_acc": round(float(np.mean(assigned_country[m] == ci)), 3)}
        return rep

    # ---- baseline: plain decode, per-image country argmax (for the acc number) ----
    base_assigned = clp.argmax(1).cpu().numpy()
    base = summarise(decode(scores), base_assigned)

    # target column marginal
    if args.marginal == "true":
        col_marg = torch.bincount(torch.tensor(true_country), minlength=N_COUNTRIES).float()
    else:
        col_marg = torch.full((N_COUNTRIES,), n_val / N_COUNTRIES)
    col_marg = col_marg.to(device)

    results = {"baseline": base, "marginal": args.marginal,
               "true_counts": col_marg.cpu().int().tolist(), "variants": {}}

    NEG = -1e4
    for temp in args.temperatures:
        for ct in args.conf_threshs:
            Q = (gated_sinkhorn(clp, col_marg, ct, temperature=temp) if ct < 1.0
                 else sinkhorn_assignment(clp, col_marg, temperature=temp))
            assigned = Q.argmax(1)
            mask = gal_country[None, :] == assigned[:, None]      # (Nval, Ngal)
            hard = summarise(decode(scores.masked_fill(~mask, NEG)), assigned.cpu().numpy())
            logQ = torch.log(Q.clamp_min(1e-9))
            soft_scores = scores + args.retrieval_weight * logQ[:, gal_country]
            soft = summarise(decode(soft_scores), assigned.cpu().numpy())
            tag = f"temp{temp:g}_cf{ct:g}"
            results["variants"][tag] = {"hard_mask": hard, "soft_bonus": soft}
            print(f"{tag}: baseline {base['median_km']:.1f} (ctry {base['country_acc']:.3f}) "
                  f"| hard {hard['median_km']:.1f} (ctry {hard['country_acc']:.3f}) "
                  f"| soft {soft['median_km']:.1f} (ctry {soft['country_acc']:.3f})", flush=True)

    print("\nPer-country (baseline vs best hard-mask variant):")
    best_tag = min(results["variants"],
                   key=lambda t: results["variants"][t]["hard_mask"]["median_km"])
    print(f"  best variant: {best_tag}")
    bh = results["variants"][best_tag]["hard_mask"]
    for cn in COUNTRIES:
        if cn in base["per_country"]:
            b, h = base["per_country"][cn], bh["per_country"][cn]
            print(f"  {cn:16s} n={b['n']:4d}  med {b['median_km']:7.1f} -> {h['median_km']:7.1f}"
                  f"   ctry_acc {b['country_acc']:.3f} -> {h['country_acc']:.3f}")

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
