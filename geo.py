"""Network, geometry, and the retrieval decoder.

Nothing in here regresses a coordinate. The net turns an image into a 192-d unit vector,
and a query gets placed wherever its nearest labelled neighbours already are.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import Dataset

R_EARTH = 6371.0088
DATA_ROOT = os.environ.get("GEO_DATA", "geo_dataset")
BACKBONE = "convnextv2_femto.fcmae_ft_in1k"


# --------------------------------------------------------------------- data

def load_labels(root=None):
    return pd.read_csv(os.path.join(root or DATA_ROOT, "train_labels.csv"))


def split_train_val(df, val_frac=0.2, seed=0):
    """(train_idx, val_idx), stratified by country. val_frac=0 puts everything in train.

    Keep the seed in step with train.py or eval.py will score the wrong images.
    """
    val = np.zeros(len(df), bool)
    if val_frac > 0:
        rng = np.random.default_rng(seed)
        for c in df.country.unique():
            idx = np.where(df.country.values == c)[0]
            rng.shuffle(idx)
            val[idx[:int(round(val_frac * len(idx)))]] = True
    return np.where(~val)[0], np.where(val)[0]


class ImageDS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), i


_cfg_cache = {}


def _data_cfg(backbone, size):
    if backbone not in _cfg_cache:
        _cfg_cache[backbone] = timm.data.resolve_model_data_config(
            timm.create_model(backbone, pretrained=False))
    cfg = dict(_cfg_cache[backbone])
    cfg["input_size"] = (3, size, size)
    cfg["crop_pct"] = 1.0
    return cfg


def eval_transform(backbone, size):
    return timm.data.create_transform(**_data_cfg(backbone, size), is_training=False)


def train_transform(backbone, size):
    # hflip=0 on purpose. Mirroring flips driving side, sign text and lane direction
    return timm.data.create_transform(**_data_cfg(backbone, size), is_training=True,
                                      auto_augment="rand-m7-n2-mstd0.5",
                                      hflip=0.0, vflip=0.0,
                                      scale=(0.5, 1.0), ratio=(0.9, 1.11))


class GeoNet(nn.Module):
    """convnextv2_femto -> GeM(learned p) -> Linear(384->192) -> L2-normalise. 4,922,161 params.

    p is learned rather than fixed so the net picks its own spot between average and max
    pooling. Landscapes want average, one distinctive building wants max.
    """

    def __init__(self, backbone=BACKBONE, dim=192, pretrained=True):
        super().__init__()
        self.bb = timm.create_model(backbone, pretrained=pretrained,
                                    num_classes=0, global_pool="")
        self.p = nn.Parameter(torch.tensor(3.0))
        self.fc = nn.Linear(self.bb.num_features, dim)

    def features(self, x):
        """Spatial feature map, B x C x L."""
        f = self.bb(x)
        return f.flatten(2) if f.ndim == 4 else f.transpose(1, 2)

    def gem(self, f):
        return f.clamp(min=1e-6).pow(self.p).mean(-1).pow(1 / self.p)

    def global_descriptor(self, f):
        """Feature map -> the 192-d unit descriptor."""
        return F.normalize(self.fc(self.gem(f)), dim=-1)

    def forward(self, x):
        return self.global_descriptor(self.features(x))

    def token_descriptors(self, f):
        """B x C x L -> B x L x dim. Same fc as the global descriptor, deliberately: reusing
        it means chamfer re-ranking costs zero extra parameters."""
        return F.normalize(self.fc(f.transpose(1, 2)), dim=-1)

    def n_params(self):
        return sum(q.numel() for q in self.parameters())


# ----------------------------------------------------------------- geometry

def haversine_km(lat1, lng1, lat2, lng2, R=R_EARTH):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    d = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def spherical_mean(lat, lng, w):
    """Weighted mean on the sphere. Average the unit vectors, not the degrees -- averaging
    degrees breaks at the antimeridian and drifts badly up near Iceland and Norway."""
    la, lo = np.radians(lat), np.radians(lng)
    x = (w * np.cos(la) * np.cos(lo)).sum(-1)
    y = (w * np.cos(la) * np.sin(lo)).sum(-1)
    z = (w * np.sin(la)).sum(-1)
    return np.degrees(np.arctan2(z, np.hypot(x, y))), np.degrees(np.arctan2(y, x))


# -------------------------------------------------------- retrieval decoder

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def l2norm(X, axis=1):
    return X / np.maximum(np.linalg.norm(X, axis=axis, keepdims=True), 1e-8)


def fit_whitening(G, dim=192, power=0.35):
    """PCA-whitening, fitted on the gallery and nothing else.

    A handful of directions carry most of the variance (vegetation, light, general scene
    look) and they swamp the small cues that actually separate two nearby places. Flatten
    them and the fine detail starts to count. power=0.35 rather than a full 0.5 whitening;
    full whitening over-amplifies the noisy tail eigenvectors.
    """
    mu = G.mean(0, keepdims=True)
    Gc = G - mu
    w, V = np.linalg.eigh(Gc.T @ Gc / len(Gc))
    o = np.argsort(-w)[:dim]
    return mu, V[:, o] @ np.diag(np.maximum(w[o], 1e-8) ** (-power))


def apply_whitening(X, mu, Wm):
    return l2norm((X - mu) @ Wm)


def csls_hubness(G, k=10):
    """Mean similarity from each gallery point to its k nearest gallery neighbours.

    subtract this from the query-gallery score and you get CSLS (Conneau et al., 2018), taken from here Bilingual lexicon induction.
    It aims to reduce hubness, whereby images from the gallery that end up nearest to every query we issue are retrieved for queries that are largely unrelated.

    Note that this is only calculated between galleries only. By doing so it is inductive (you only compute once for forever without seeing any gallery images
    paired with query images again during index search, so the images themselves can "vote" on "hubness") and performed best, as opposed to the
    transductive version (calculating based upon query batches).
    """
    Gt = torch.from_numpy(G).to(_device())
    S = Gt @ Gt.T
    S.fill_diagonal_(-1e4)
    return S.topk(k, dim=0).values.mean(0).cpu().numpy()


def retrieve(Q, G, hub=None, beta=0.25, topk=64):
    """Top-k gallery neighbours per query, by cosine minus the CSLS hubness term."""
    dev = _device()
    S = torch.from_numpy(Q).to(dev) @ torch.from_numpy(G).to(dev).T
    if hub is not None and beta:
        S = S - beta * torch.from_numpy(hub).to(dev)[None, :]
    v, i = S.topk(topk, dim=1)
    return v.cpu().numpy(), i.cpu().numpy()


def chamfer_rerank(qtok, gtok, idx, sim, lam=3.0, chunk=8):
    """Re-score the shortlist on local token matching.

        cham(q,g) = 0.5 * ( mean_i max_j <q_i,g_j> + mean_j max_i <q_i,g_j> )
        score     = global_similarity + lam * cham(q,g)

    The global descriptor smushes the entire feature map into a single vector and so two images 
    may be near each other without relating in any particular way. Token sets ask a stronger question. 

    Is some patch of this image another patch from that one? That distinguishes "another road in the same country" from "the same street".

    qtok (nq, L, d), gtok (ng, L, d), per-token L2-normalised. idx (nq, K) is the shortlist,
    sim (nq, K) its global scores. Returns both re-sorted.
    """
    dev = _device()
    qt = torch.as_tensor(qtok).to(dev)
    gt = torch.as_tensor(gtok).to(dev)
    ii = torch.as_tensor(idx.astype(np.int64)).to(dev)
    cham = torch.empty(idx.shape, dtype=torch.float32, device=dev)
    for s in range(0, len(idx), chunk):
        e = min(s + chunk, len(idx))
        q = qt[s:e].float()                            
        c = gt[ii[s:e]].float()                        
        m = torch.einsum("bld,bkmd->bklm", q, c)
        cham[s:e] = 0.5 * (m.max(3).values.mean(2) + m.max(2).values.mean(2))
    score = sim + lam * cham.cpu().numpy()
    o = np.argsort(-score, axis=1)
    return np.take_along_axis(score, o, 1), np.take_along_axis(idx, o, 1)


def local_mode_decode(sim, idx, lat, lng, radius_km=60.0, temp=0.1):
    """Similarity-weighted spherical mean, but only over neighbours within radius_km of the
    top-1.

    The radius is everything. Average over the stuff we find and there is one solid neighbor
    at the correct street that is overruled by one statistically insignificant point further 
    down the country so the actual answer is lying in some field between the two of us.
    """
    gl, gn = lat[idx], lng[idx]
    d1 = haversine_km(gl[:, :1], gn[:, :1], gl, gn)
    w = (d1 < radius_km) * np.exp((sim - sim[:, :1]) / temp)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return spherical_mean(gl, gn, w)
