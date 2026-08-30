"""
T6/R8 -- cache the 4-model teacher ensemble's country probabilities over a fold's TRAIN set,
for distillation into a single compliant student (Further_Analysis_&_Findings §4.5).

The score-space ensemble (E3e + E4 + E3d + E3h) reaches 122.3 km CV vs the best single model's
129 -- and its gain (per T6_FINDINGS) is mostly killing catastrophic wrong-country picks
(mean 427 -> 368; FR median 451 -> 373, ES 312 -> 216). Its country posterior is the
transferable part: soft, better-calibrated, architecture-agnostic. Cache it once here; the
student trains with a KL term toward it.

Output: outputs/experiments/teacher_country_fold{f}.npz with `filenames` (str array) and
`probs` (N, 12) float32, the flip-TTA mean softmax across the 4 teachers.

Usage:
    python -m src.cache_teacher_country --fold 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from timm.data import resolve_data_config
import timm
from torch.utils.data import DataLoader

from src.data.dataset import GeoDataset, load_fold_cells_multi, load_labelled_df
from src.data.multiview import GlobalLocalTransform
from src.geo.countries import COUNTRY_TO_IDX
from src.ensemble_cv import _infer_arch
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]

TEACHERS = [
    ("outputs/experiments/T6_E3e_cells368_hsmooth45_fold0/fold0/best_model.pt", "geo_cells_multi_fine"),
    ("outputs/experiments/T6_E3d_cells192_hsmooth65_fold0/fold0/best_model.pt", "geo_cells_multi_mid"),
    ("failed_experiments/T6_E3h_tightcrops_neutral/run/fold0/best_model.pt", "geo_cells_multi_fine"),
    ("outputs/experiments/T6_E4_geoclip_fold0/fold0/best_model.pt", "geo_cells_multi_fine"),
]
BACKBONE = "efficientnet_b0.ra4_e3600_r224_in1k"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--input-size", type=int, default=288)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_labelled_df()
    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    filenames = train_df["filename"].tolist()

    probe = timm.create_model(BACKBONE, pretrained=False)
    cfg = resolve_data_config({}, model=probe)
    del probe
    tf = GlobalLocalTransform(args.input_size, cfg["mean"], cfg["std"],
                              float(cfg.get("crop_pct", 0.9)), False, 0.64)

    prob_sum = np.zeros((len(filenames), 12), dtype=np.float64)
    for ckpt, cells_sub in TEACHERS:
        sd = torch.load(REPO_ROOT / ckpt, map_location=device)
        n_part, emb, proj, chh, loc = _infer_arch(sd)
        cell_info = load_fold_cells_multi(
            args.fold, n_part,
            cells_dir=REPO_ROOT / "outputs" / "protocol" / cells_sub)
        m = MultiViewRetrievalGeoClassifier(
            BACKBONE, cell_info["n_cells"], cell_info["cell_to_country"],
            embedding_dim=emb, pretrained=False, dropout=0.3,
            cell_proj_dim=(proj or None), use_location_encoder=loc,
            country_head_hidden=chh).to(device)
        m.load_state_dict(sd)
        m.eval()
        ds = GeoDataset(train_df, train=False, input_size=args.input_size,
                        multi_cell_info=cell_info, fix_orientation=True)
        ds.transform = tf
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        rows = []
        for b in dl:
            img = b["image"].to(device, non_blocking=True)
            _, _, ctry, _, _ = m(img)
            _, _, ctry2, _, _ = m(torch.flip(img, dims=[-1]))
            p = 0.5 * (F.softmax(ctry.float(), -1) + F.softmax(ctry2.float(), -1))
            rows.append(p.cpu().numpy())
        prob_sum += np.concatenate(rows)
        print(f"cached {ckpt.split('/')[-3]}", flush=True)

    probs = (prob_sum / len(TEACHERS)).astype(np.float32)
    acc = float(np.mean(probs.argmax(1) == train_df["country"].map(COUNTRY_TO_IDX).to_numpy()))
    out = REPO_ROOT / "outputs" / "experiments" / f"teacher_country_fold{args.fold}.npz"
    np.savez_compressed(out, filenames=np.array(filenames), probs=probs)
    print(f"\nteacher ensemble country train-acc = {acc:.3f}   (N={len(filenames)})")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
