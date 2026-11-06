"""Descriptor extraction, with multi-scale test-time augmentation.

The same network is run at 448, 512, and 576 and the descriptors averaged. The descriptors are also normalized 
by dividing through by the L2 norm. The network uses the same weights for all these images, thus there are no added 
parameters here. 

We train at 448, and when the larger inputs are fed in the same network allows GeM to average on a 
higher quality spatial grid, producing finer grained detail the descriptor is trained on.

For the respective worth of the decode stage see the ablation ladder in the README.md (the measured numbers are reported there).

The 448 px pass also writes the token-based descriptors, which is needed for chamfer re-ranking in predict.py and eval.py.
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import geo

SIZES = [448, 512, 576]
TOKEN_SIZE = 448


def descriptors(net, paths, backbone, sizes, token_size, bs=48, workers=8):
    """One global descriptor per size, plus spatial tokens at token_size only."""
    net.eval()
    out, tokens = {}, None
    for size in sizes:
        loader = DataLoader(geo.ImageDS(paths, geo.eval_transform(backbone, size)),
                            batch_size=bs, num_workers=workers, pin_memory=True)
        glob, tok = [], []
        with torch.no_grad():
            for x, _ in loader:
                x = x.cuda()
                with torch.autocast("cuda", torch.float16):
                    f = net.features(x)
                    glob.append(net.global_descriptor(f).float().cpu())
                if size == token_size:
                    # fp32 here, outside the autocast block. Re-ranking turns on small score
                    # gaps, and a half-precision wobble is enough to swap two shortlist
                    # entries and move the prediction a few hundred km. Not worth the speed.
                    tok.append(net.token_descriptors(f.float()).half().cpu())
        out[size] = torch.cat(glob).numpy().astype(np.float16)
        if tok:
            tokens = torch.cat(tok).numpy()
        print(f"  {size} px: {out[size].shape}"
              + (f" + tokens {tokens.shape}" if tok else ""), flush=True)
    return out, tokens


def combine(emb, sizes):
    """The TTA step itself: mean of the per-scale descriptors, back onto the unit sphere."""
    return geo.l2norm(sum(emb[s].astype(np.float32) for s in sizes))


def image_list(data, split):
    if split == "train":
        names = list(geo.load_labels(data).filename)
    else:
        # Filter by extension. listdir happily returns .DS_Store and friends, and the
        # submission has to be exactly 2,400 rows -- one stray file and it is rejected.
        names = sorted(f for f in os.listdir(os.path.join(data, "holdout_public"))
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
    folder = "train" if split == "train" else "holdout_public"
    return names, [os.path.join(data, folder, f) for f in names]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["train", "holdout"], default="train")
    p.add_argument("--backbone", default=geo.BACKBONE)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--data", default=geo.DATA_ROOT)
    p.add_argument("--bs", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()

    names, paths = image_list(a.data, a.split)
    net = geo.GeoNet(a.backbone, a.dim, pretrained=False).cuda()
    net.load_state_dict(torch.load(a.ckpt, map_location="cuda"))
    print(f"{a.ckpt}: {net.n_params()} params, {len(paths)} images", flush=True)

    emb, tokens = descriptors(net, paths, a.backbone, SIZES, TOKEN_SIZE, a.bs, a.workers)
    np.savez(a.out, names=np.array(names), tokens=tokens,
             **{f"emb{s}": v for s, v in emb.items()})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
