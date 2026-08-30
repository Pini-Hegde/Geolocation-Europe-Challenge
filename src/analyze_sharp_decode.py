"""
Does a SHARP near-nearest-neighbour retrieval decode beat the mode (spherical-mean) decode?

Our CV folds separate 500m location clusters, so on CV a val image's nearest gallery point
is >=500m away and usually much more -- the sharp decode looks bad. But the holdout is
almost certainly spatially dense w.r.t. training (34% of train points have a <1km neighbour;
the challenge does not promise train/test separation). This script:

  1. reports, on a fold's val set, the top-1 retrieval cosine-similarity distribution and
     mode-vs-sharp decode error BUCKETED by that similarity -- so we can pick a similarity
     threshold above which sharp decode already wins even with the 500m CV floor;
  2. reports the same similarity distribution on the holdout -- so we know what fraction of
     the 2400 holdout images clear that threshold (an estimate of the sharp-decode payoff).

Usage:
    python -m src.analyze_sharp_decode --checkpoint <ckpt> --fold 0 \
        --cells-dir outputs/protocol/geo_cells_multi_fine --n-partitions 4 \
        --embedding-dim 224 --cell-proj-dim 224
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import GeoDataset, load_labelled_df
from src.data.multiview import GlobalLocalTransform
from src.geo.mode_retrieval import spherical_mode_decode
from src.geo.orientation import fix_orientation as _fix_orientation
from src.geo.spherical import haversine_km_np
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier
from src.train_multiview_retrieval_geo import hierarchy_scores, make_multiview_loaders
from src.ensemble_cv import _infer_arch

REPO_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_DIR = REPO_ROOT / "geo_dataset" / "holdout_public"


class _Holdout(Dataset):
    def __init__(self, names, tf):
        self.names, self.tf = names, tf

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = _fix_orientation(Image.open(HOLDOUT_DIR / n).convert("RGB"), n)
        return {"image": self.tf(img), "filename": n}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cells-dir", default="outputs/protocol/geo_cells_multi_fine")
    ap.add_argument("--n-partitions", type=int, default=4)
    ap.add_argument("--embedding-dim", type=int, default=224)
    ap.add_argument("--cell-proj-dim", type=int, default=224)
    ap.add_argument("--retrieval-weight", type=float, default=16.0)
    ap.add_argument("--radius-km", type=float, default=200.0)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sd = torch.load(args.checkpoint, map_location=dev)
    n_part, emb, proj, chh, loc = _infer_arch(sd)
    la = SimpleNamespace(
        backbone="efficientnet_b0.ra4_e3600_r224_in1k", folds=[args.fold],
        n_partitions=n_part, cells_dir=str(REPO_ROOT / args.cells_dir),
        input_size=288, local_fraction=0.64, batch_size=24, samples_per_cell=4,
        eval_batch_size=24, num_workers=8, embedding_dim=emb, dropout=0.3,
        fix_orientation=True, label_smooth_tau=0.0)
    _, gl, vl, cells, _ = make_multiview_loaders(la, args.fold)
    m = MultiViewRetrievalGeoClassifier(
        la.backbone, cells["n_cells"], cells["cell_to_country"], embedding_dim=emb,
        pretrained=False, dropout=0.3, cell_proj_dim=(proj or None),
        use_location_encoder=loc, country_head_hidden=chh).to(dev)
    m.load_state_dict(sd)
    m.eval()

    gemb, gctry, gcell, guv = [], [], [], []
    for b in gl:
        cl, off, ctry, e, vw = m(b["image"].to(dev))
        gemb.append(e); gctry.append(b["country_idx"].to(dev))
        gcell.append(b["cell_ids"].to(dev)); guv.append(b["target_vec"].to(dev))
    gal = {"embeddings": torch.cat(gemb), "country": torch.cat(gctry),
           "cell_ids": torch.cat(gcell),
           "unit_vectors": F.normalize(torch.cat(guv), dim=-1)}

    sims1, e_mode, e_sharp, tctry = [], [], [], []
    for b in vl:
        cl, off, ctry, e, vw = m(b["image"].to(dev))
        cos = e.float() @ gal["embeddings"].float().T
        h = hierarchy_scores(m, ctry, cl, gal)
        scores = h + args.retrieval_weight * cos
        tl, tn = b["lat"].numpy(), b["lng"].numpy()
        mlat, mlng, _ = spherical_mode_decode(scores, gal["unit_vectors"], top_m=100,
                                              radius_km=args.radius_km, kernel="hard")
        i1 = scores.argmax(-1)
        nnv = gal["unit_vectors"][i1]
        slat = torch.rad2deg(torch.asin(nnv[:, 2].clamp(-1, 1))).cpu().numpy()
        slng = torch.rad2deg(torch.atan2(nnv[:, 1], nnv[:, 0])).cpu().numpy()
        sims1.append(cos.max(-1).values.cpu().numpy())
        e_mode.append(haversine_km_np(tl, tn, mlat.cpu().numpy(), mlng.cpu().numpy()))
        e_sharp.append(haversine_km_np(tl, tn, slat, slng))
        tctry.append(b["country_idx"].numpy())
    sims1 = np.concatenate(sims1); e_mode = np.concatenate(e_mode)
    e_sharp = np.concatenate(e_sharp)

    print(f"val fold {args.fold}: n={len(sims1)}")
    print(f"top-1 retrieval cosine sim: p50={np.percentile(sims1,50):.3f} "
          f"p90={np.percentile(sims1,90):.3f} p99={np.percentile(sims1,99):.3f} "
          f"max={sims1.max():.3f}")
    print(f"\noverall: mode median {np.median(e_mode):.1f}  sharp median {np.median(e_sharp):.1f}")
    print("\nby top-1 sim bucket  (n, mode med, sharp med, sharp wins?):")
    for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
        msk = (sims1 >= lo) & (sims1 < hi)
        if msk.sum() < 5:
            continue
        mm, sm = np.median(e_mode[msk]), np.median(e_sharp[msk])
        print(f"  [{lo:.2f},{hi:.2f})  n={msk.sum():4d}  mode {mm:6.1f}  sharp {sm:6.1f}  "
              f"{'SHARP' if sm < mm else 'mode'}")

    # hybrid: sharp where sim >= t, mode elsewhere -- sweep t
    print("\nhybrid median by threshold t (sharp if sim>=t):")
    for t in [1.01, .95, .9, .85, .8, .75, .7, .6]:
        use = sims1 >= t
        hyb = np.where(use, e_sharp, e_mode)
        print(f"  t={t:.2f}  {use.mean()*100:5.1f}% sharp  ->  median {np.median(hyb):6.1f}  "
              f"mean {hyb.mean():6.0f}")

    # holdout similarity distribution
    from timm import create_model
    from timm.data import resolve_data_config
    probe = create_model(la.backbone, pretrained=False)
    cfg = resolve_data_config({}, model=probe); del probe
    tf = GlobalLocalTransform(288, cfg["mean"], cfg["std"], float(cfg.get("crop_pct", .9)),
                              False, 0.64)
    names = sorted(p.name for p in HOLDOUT_DIR.glob("*.jpg"))
    hl = DataLoader(_Holdout(names, tf), batch_size=24, num_workers=8, pin_memory=True)
    hs = []
    for b in hl:
        cl, off, ctry, e, vw = m(b["image"].to(dev))
        hs.append((e.float() @ gal["embeddings"].float().T).max(-1).values.cpu().numpy())
    hs = np.concatenate(hs)
    print(f"\nHOLDOUT top-1 retrieval cosine sim (n={len(hs)}):")
    print(f"  p50={np.percentile(hs,50):.3f} p75={np.percentile(hs,75):.3f} "
          f"p90={np.percentile(hs,90):.3f} p95={np.percentile(hs,95):.3f} max={hs.max():.3f}")
    for t in [.95, .9, .85, .8, .75, .7]:
        print(f"  holdout fraction with sim >= {t}: {np.mean(hs >= t)*100:.1f}%")


if __name__ == "__main__":
    main()
