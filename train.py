"""Train the embedding network.

The loss. Take a batch of B images, compute every pairwise great-circle distance d_ij:

    Q = softmax_j( -d_ij / tau_km )        tau_km = 15    (what geography says)
    P = softmax_j( (e_i . e_j) / temp )    temp   = 0.07  (what the embedding says)
    loss = KL(Q || P)                      diagonal masked out of both

Each picture of the batch would then be a positive, weighted according to how close to the ground truth it is. 
If there is a "hard" positive/negative split at a certain, constant radius, that gradient is lost.

When changing a value in here, have it update tau_km. An old version used 250 km, 
when the target couldn't distinguish 1 km from 40 km, and so the embedding, naturally, 
learned which of the regions the photo was extracted from, but not place at all. That's the only constant used. It separates 126 km of plateau from the rest.

The sampler. Only about a quarter of the images have any partner within 500 m, so a random
batch usually contains no positive pair at all and the loss goes nowhere. Batches are built
as 12 anchors x 4 images from within 25 km of that anchor. Bonus: the other three images in
a cluster make excellent hard negatives.

    --val-frac 0.2    hold out a stratified 20%, report retrieval error on it
    --val-frac 0      train on all 11,758; this is the model that ships
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from scipy.spatial import cKDTree
import geo


class GeoBatchSampler(Sampler):
    """Each batch is `clusters` anchors x `per_cluster` images from within neighbour_km."""

    def __init__(self, neighbours, clusters, per_cluster, n_batches, seed=0):
        self.nb = neighbours
        self.clusters, self.per_cluster, self.n_batches = clusters, per_cluster, n_batches
        self.rng = np.random.default_rng(seed)  # one rng for the run, so --seed reproduces

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = self.rng
        for _ in range(self.n_batches):
            b = []
            for _ in range(self.clusters):
                pool = self.nb[rng.integers(len(self.nb))]
                b += rng.choice(pool, size=self.per_cluster,
                                replace=len(pool) < self.per_cluster).tolist()
            yield b


def neighbour_pools(lat, lng, train_idx, radius_km):
    """For each training image, which other training images sit within radius_km.

    Done on the unit sphere with a chord-length cutoff so cKDTree can do it in euclidean
    space. Val images are filtered out here, not later -- they must never enter a batch.
    """
    la, lo = np.radians(lat), np.radians(lng)
    xyz = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
    chord = 2 * np.sin(radius_km / (2 * geo.R_EARTH))
    keep = set(train_idx.tolist())
    return [np.array([j for j in n if j in keep])
            for n in cKDTree(xyz).query_ball_point(xyz[train_idx], chord)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--backbone", default=geo.BACKBONE)
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--tau-km", type=float, default=15.0)
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--clusters-per-batch", type=int, default=12)
    p.add_argument("--imgs-per-cluster", type=int, default=4)
    p.add_argument("--neighbour-km", type=float, default=25.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--data", default=geo.DATA_ROOT)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    df = geo.load_labels(a.data)
    lat, lng = df.lat.values, df.lng.values
    tr, va = geo.split_train_val(df, a.val_frac, a.seed)
    print(f"train={len(tr)} val={len(va)}", flush=True)

    paths = [os.path.join(a.data, "train", f) for f in df.filename]
    pools = neighbour_pools(lat, lng, tr, a.neighbour_km)
    batch = a.clusters_per_batch * a.imgs_per_cluster
    n_batches = max(1, len(tr) // batch)

    net = geo.GeoNet(a.backbone, a.dim, pretrained=True).cuda()
    n_params = net.n_params()
    print(f"{a.backbone}: {n_params} params, compliant={n_params <= 5_000_000}", flush=True)

    opt = torch.optim.AdamW(
        [{"params": net.bb.parameters(), "lr": a.lr / 3},
         {"params": [net.p] + list(net.fc.parameters()), "lr": a.lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, [a.lr / 3, a.lr], total_steps=a.epochs * n_batches, pct_start=0.15)
    scaler = torch.amp.GradScaler()
    loader = DataLoader(geo.ImageDS(paths, geo.train_transform(a.backbone, a.size)),
                        batch_sampler=GeoBatchSampler(pools, a.clusters_per_batch,
                                                      a.imgs_per_cluster, n_batches, a.seed),
                        num_workers=a.workers, pin_memory=True, persistent_workers=True)

    @torch.no_grad()
    def embed(idx):
        net.eval()
        ds = geo.ImageDS([paths[i] for i in idx], geo.eval_transform(a.backbone, a.size))
        out = []
        for x, _ in DataLoader(ds, batch_size=48, num_workers=a.workers, pin_memory=True):
            with torch.autocast("cuda", torch.float16):
                out.append(net(x.cuda()).float().cpu())
        return torch.cat(out).numpy()

    def val_error():
        """Rough top-1 error, single scale, no decoding. This is a progress light, not a
        result -- the number worth quoting comes from eval.py with the full decoder."""
        _, i = geo.retrieve(embed(va), embed(tr), topk=1)
        g = tr[i[:, 0]]
        return geo.haversine_km(lat[va], lng[va], lat[g], lng[g])

    history = []
    for ep in range(a.epochs):
        net.train()
        total = n = 0
        for x, j in loader:
            x = x.cuda(non_blocking=True)
            j = j.numpy()
            d = geo.haversine_km(lat[j][:, None], lng[j][:, None],
                                 lat[j][None, :], lng[j][None, :])
            mask = 1e4 * torch.eye(len(j), device="cuda")
            q = torch.softmax(-torch.from_numpy(d).cuda().float() / a.tau_km - mask, dim=1)
            with torch.autocast("cuda", torch.float16):
                e = net(x)
                s = e @ e.T / a.temp - mask
                # KL(Q||P) = sum_j q log q - sum_j q log p. The q log q half has no weights
                # in it, so it shifts the printed loss but contributes nothing to the
                # gradient. Kept anyway so the number is a real KL and not a proxy.
                loss = (q * (torch.log(q.clamp_min(1e-12))
                             - torch.log_softmax(s, dim=1))).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            n += 1

        line = f"ep{ep:3d} loss {total / n:.4f}"
        if len(va) and (ep % 4 == 3 or ep == a.epochs - 1):
            d = val_error()
            line += f" | val top-1 median {np.median(d):7.2f} mean {d.mean():7.1f}"
        print(line, flush=True)
        history.append(line)
        # Last epoch wins, always. The budget is fixed at 40 up front and there is no "best
        # checkpoint" logic anywhere, on purpose -- picking the best epoch by val score is
        # model selection on the val split, and the reported number stops being honest.
        torch.save(net.state_dict(), os.path.join(a.out, "final.pt"))

    json.dump({"args": vars(a), "params": n_params, "history": history},
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(f"done: {a.out}, {n_params} params", flush=True)


if __name__ == "__main__":
    main()
