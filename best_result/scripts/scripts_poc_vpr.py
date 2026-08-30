"""PoC: train a <=5M-param backbone as a FINE-GRAINED place-recognition embedding
(positives = geographically co-located images), validated on an IID 200/country split
(matching how holdout_public appears to be constructed) AND on the project's 500m-group
fold 0, so the two protocols can be compared directly."""
import os, json, math, argparse, numpy as np, pandas as pd, torch, timm
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from scipy.spatial import cKDTree

ROOT = "/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"; D = ROOT + "/geo_dataset"
R = 6371.0088
p = argparse.ArgumentParser()
p.add_argument("--backbone", default="efficientnet_b0.ra4_e3600_r224_in1k")
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--size", type=int, default=320)
p.add_argument("--dim", type=int, default=256)
p.add_argument("--tau-km", type=float, default=15.0)   # geo softness of the target
p.add_argument("--temp", type=float, default=0.07)     # embedding softmax temperature
p.add_argument("--clusters-per-batch", type=int, default=12)
p.add_argument("--imgs-per-cluster", type=int, default=4)
p.add_argument("--neighbor-km", type=float, default=25.0)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--split", default="official", choices=["iid","fold0","official"])
p.add_argument("--fold", type=int, default=0)
p.add_argument("--tag", default="poc")
a = p.parse_args()

df = pd.read_csv(D + "/train_labels.csv")
lat, lng, cty = df.lat.values, df.lng.values, df.country.values
def xyz(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.c_[np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]
P = xyz(lat, lng); tree = cKDTree(P)
def hav(a1,b1,a2,b2):
    a1,b1,a2,b2 = map(np.radians,(a1,b1,a2,b2))
    h = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin((b2-b1)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))

if a.split == "iid":
    rng = np.random.default_rng(0)
    va = np.concatenate([rng.permutation(np.where(cty==c)[0])[:200] for c in np.unique(cty)])
elif a.split == "official":
    f = json.load(open(ROOT+"/outputs/protocol/folds/folds_official.json"))["filename_to_fold"]
    va = np.where(np.array([f[x] for x in df.filename])==a.fold)[0]
else:
    f = json.load(open(ROOT+"/outputs/protocol/folds/folds.json"))["filename_to_fold"]
    va = np.where(np.array([f[x] for x in df.filename])==0)[0]
m = np.ones(len(df), bool); m[va] = False; tr = np.where(m)[0]
print(f"split={a.split} train={len(tr)} val={len(va)}", flush=True)

# --- geo-aware batch sampler: each batch = C anchor points, each with N images within neighbor_km
chord = 2*np.sin(a.neighbor_km/(2*R))
nbrs = tree.query_ball_point(P[tr], chord)
tr_set = set(tr.tolist())
nbrs = [np.array([j for j in n if j in tr_set]) for n in nbrs]
class GeoBatchSampler(Sampler):
    def __init__(s, n_batches): s.nb = n_batches
    def __len__(s): return s.nb
    def __iter__(s):
        rng = np.random.default_rng()
        for _ in range(s.nb):
            b = []
            for _ in range(a.clusters_per_batch):
                k = rng.integers(len(tr)); pool = nbrs[k]
                pick = rng.choice(pool, size=a.imgs_per_cluster, replace=len(pool) < a.imgs_per_cluster)
                b += pick.tolist()
            yield b
BS = a.clusters_per_batch * a.imgs_per_cluster
n_batches = max(1, len(tr) // BS)

cfg = timm.data.resolve_model_data_config(timm.create_model(a.backbone, pretrained=False))
cfg["input_size"] = (3, a.size, a.size); cfg["crop_pct"] = 1.0
tf_tr = timm.data.create_transform(**cfg, is_training=True, auto_augment="rand-m7-n2-mstd0.5",
                                   hflip=0.0, vflip=0.0, scale=(0.5, 1.0), ratio=(0.9, 1.11))
tf_ev = timm.data.create_transform(**cfg, is_training=False)
class DS(Dataset):
    def __init__(s, idx, tf): s.idx, s.tf = idx, tf
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        j = s.idx[i]
        return s.tf(Image.open(D+"/train/"+df.filename.iloc[j]).convert("RGB")), j
class BDS(Dataset):   # sampler yields absolute indices
    def __len__(s): return len(df)
    def __getitem__(s, j):
        return tf_tr(Image.open(D+"/train/"+df.filename.iloc[j]).convert("RGB")), j

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.bb = timm.create_model(a.backbone, pretrained=True, num_classes=0, global_pool="")
        c = s.bb.num_features
        s.p = nn.Parameter(torch.tensor(3.0)); s.fc = nn.Linear(c, a.dim)
    def forward(s, x):
        f = s.bb(x)
        f = f.flatten(2) if f.ndim == 4 else f.transpose(1, 2)
        g = f.clamp(min=1e-6).pow(s.p).mean(-1).pow(1/s.p)
        return F.normalize(s.fc(g), dim=-1)
net = Net().cuda()
npar = sum(q.numel() for q in net.parameters())
print(f"backbone={a.backbone} params={npar/1e6:.3f}M  compliant={npar<=5_000_000}", flush=True)

opt = torch.optim.AdamW([{"params": net.bb.parameters(), "lr": a.lr/3},
                         {"params": [net.p]+list(net.fc.parameters()), "lr": a.lr}], weight_decay=0.01)
steps = a.epochs*n_batches
sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr/3, a.lr], total_steps=steps, pct_start=0.15)
scaler = torch.amp.GradScaler()
dl_tr = DataLoader(BDS(), batch_sampler=GeoBatchSampler(n_batches), num_workers=10, pin_memory=True)

@torch.no_grad()
def embed(idx):
    net.eval(); o = []
    for x, _ in DataLoader(DS(idx, tf_ev), batch_size=64, num_workers=10, pin_memory=True):
        with torch.autocast("cuda", torch.float16): o.append(net(x.cuda()).float().cpu())
    return torch.cat(o)
def val():
    Et, Ev = embed(tr).cuda(), embed(va).cuda()
    v, i = (Ev @ Et.T).topk(8, dim=1); gi = tr[i.cpu().numpy()]
    d = hav(lat[va], lng[va], lat[gi[:,0]], lng[gi[:,0]])
    return np.median(d), d.mean(), (d < 25).mean(), (d < 200).mean(), (d < 750).mean()

best = 1e9
for ep in range(a.epochs):
    net.train()
    for x, j in dl_tr:
        x = x.cuda(non_blocking=True); j = j.numpy()
        dm = hav(lat[j][:,None], lng[j][:,None], lat[j][None,:], lng[j][None,:])
        q = torch.from_numpy(dm).cuda().float()
        q = torch.softmax(-q/a.tau_km - 1e4*torch.eye(len(j), device="cuda"), dim=1)
        with torch.autocast("cuda", torch.float16):
            e = net(x)
            s = e @ e.T / a.temp
            s = s - 1e4*torch.eye(len(j), device="cuda")
            loss = -(q * torch.log_softmax(s, dim=1)).sum(1).mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    if ep % 3 == 2 or ep == a.epochs-1:
        md, mn, r25, r200, r750 = val()
        print(f"ep{ep:3d} loss {loss.item():.3f} | VAL[{a.split}] median {md:7.2f} mean {mn:7.1f} "
              f"<25 {r25:.3f} <200 {r200:.3f} <750 {r750:.3f}", flush=True)
        if md < best:
            best = md
            os.makedirs(f"{ROOT}/outputs/poc/{a.tag}", exist_ok=True)
            torch.save(net.state_dict(), f"{ROOT}/outputs/poc/{a.tag}/best.pt")
            np.savez(f"{ROOT}/outputs/poc/{a.tag}/emb.npz",
                     val_idx=va, tr_idx=tr,
                     val=embed(va).numpy().astype(np.float16),
                     gal=embed(tr).numpy().astype(np.float16))
print(f"BEST[{a.tag}, split={a.split}] median {best:.2f} km  params {npar/1e6:.3f}M", flush=True)
