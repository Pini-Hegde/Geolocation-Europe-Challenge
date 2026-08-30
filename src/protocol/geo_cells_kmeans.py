"""
K-means geo-cells (T6/E7) -- an alternative to the recursive kd-tree median-split cells.

Motivation: kd-tree cells are axis-aligned rectangles built by splitting at the median of
the wider axis. A leaf can be a long thin strip that spans, say, a town and the countryside
beyond it -- two visually different places in one class. PIGEON (Haas et al. 2024) argues
visually-/semantically-coherent cells classify far better; lacking admin boundaries (external
data), balanced k-means on the training coordinates is the compliant approximation: each
cell is a compact blob around a local density peak.

Per country, MiniBatchKMeans on the (equal-earth-projected) training coordinates with
`k = round(n_country / target_per_cell)`. Cell anchor = spherical mean of the cell's training
points (not the k-means centroid). Assignment = nearest cell anchor (great-circle), for train,
val and holdout alike. K independent partitions via different k-means seeds, same as
geo_cells_multi.

Usage:
    python -m src.protocol.geo_cells_kmeans --folds 0 --target-per-cell 30 \
        --out-dir outputs/protocol/geo_cells_kmeans_t30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from src.geo.countries import COUNTRY_TO_IDX
from src.geo.spherical import (
    haversine_km_np,
    latlon_to_unit_vector_np,
    unit_vector_to_latlon_np,
)
from src.protocol.geo_cells import load_joined, N_FOLDS

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "protocol" / "geo_cells_kmeans"
SEED_BASE = 3000


def _eep(lat, lng):
    """crude equal-earth-ish projection for clustering (km-scaled local tangent)."""
    return np.stack([np.radians(lng) * np.cos(np.radians(lat)) * 6371.0,
                     np.radians(lat) * 6371.0], axis=-1)


def _spherical_mean(lat, lng):
    v = latlon_to_unit_vector_np(lat, lng).mean(axis=0)
    v = v / np.linalg.norm(v)
    la, lo = unit_vector_to_latlon_np(v[None, :])
    return float(la[0]), float(lo[0])


def build_partition(df_train: pd.DataFrame, df_all: pd.DataFrame, target_per_cell: int,
                    seed: int) -> dict:
    cells = {}
    anchors_lat, anchors_lng, anchor_country = [], [], []
    cid = 0
    for country, g in df_train.groupby("country"):
        n = len(g)
        k = max(1, round(n / target_per_cell))
        xy = _eep(g["lat"].to_numpy(), g["lng"].to_numpy())
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=256)
        lab = km.fit_predict(xy)
        for c in range(k):
            m = lab == c
            if m.sum() == 0:
                continue
            alat, alng = _spherical_mean(g["lat"].to_numpy()[m], g["lng"].to_numpy()[m])
            cells[str(cid)] = {"centroid_lat": alat, "centroid_lng": alng,
                               "country": country, "n_train": int(m.sum())}
            anchors_lat.append(alat); anchors_lng.append(alng)
            anchor_country.append(country)
            cid += 1
    anchors_lat = np.asarray(anchors_lat); anchors_lng = np.asarray(anchors_lng)
    anchor_country = np.asarray(anchor_country)

    # assign every row (train+val+holdout share the scheme) to the nearest same-country anchor
    assignments = {}
    for country, g in df_all.groupby("country"):
        mask = anchor_country == country
        idxs = np.where(mask)[0]
        al, an = anchors_lat[mask], anchors_lng[mask]
        for fn, la, lo in zip(g["filename"], g["lat"].to_numpy(), g["lng"].to_numpy()):
            d = haversine_km_np(la, lo, al, an)
            assignments[fn] = {"cell_id": int(idxs[int(d.argmin())])}
    return {"meta": {"n_cells": cid, "seed": seed, "target_per_cell": target_per_cell,
                     "method": "per_country_minibatch_kmeans_eep"},
            "cells": cells, "assignments": assignments}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)))
    ap.add_argument("--n-partitions", type=int, default=4)
    ap.add_argument("--target-per-cell", type=int, default=30)
    ap.add_argument("--region-target", type=int, default=0,
                    help="if >0, also emit fold{f}_region_cells.json -- a coarse single "
                         "k-means partition with ~this many images per region (Muller-Budack "
                         "country->region->cell hierarchy level). ~150-250 gives ~48 regions.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--full-data", action="store_true",
                    help="also build a fold -1 (all-data) scheme for the final model")
    args = ap.parse_args()
    df = load_joined()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds = list(args.folds) + ([-1] if args.full_data else [])
    for fold in folds:
        train = df if fold < 0 else df[df["fold"] != fold]
        for kp in range(args.n_partitions):
            part = build_partition(train, df, args.target_per_cell, SEED_BASE + kp)
            (args.out_dir / f"fold{fold}_p{kp}_cells.json").write_text(json.dumps(part))
            sizes = np.array([c["n_train"] for c in part["cells"].values()])
            print(f"fold {fold} p{kp}: {part['meta']['n_cells']} cells "
                  f"(train/cell min {sizes.min()} med {int(np.median(sizes))} max {sizes.max()})")
        if args.region_target > 0:
            reg = build_partition(train, df, args.region_target, SEED_BASE + 99)
            reg["meta"]["n_cells"] = reg["meta"]["n_cells"]  # load_fold_region reads meta.n_cells
            (args.out_dir / f"fold{fold}_region_cells.json").write_text(json.dumps(reg))
            print(f"fold {fold} regions: {reg['meta']['n_cells']}")
    print(f"written to {args.out_dir}/")


if __name__ == "__main__":
    main()
