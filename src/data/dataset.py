"""
Dataset for the geolocation task: loads an image from `geo_dataset/train/`, its
(lat, lng) label from `train_labels.csv`, and hands back the spherical-vector target
(`src/geo/spherical.py`) rather than raw degrees -- every training script should build
targets through this class, not by re-deriving the conversion inline.

Fold membership comes from the fixed `outputs/protocol/folds/folds.json` (group-stratified
5-fold, Experiment_Design.md §2) -- `GeoDataset(..., folds=[...])` selects the rows whose
fold id is in that list, so one fold held out gives the validation set and the rest give
training, per the two-stage screening protocol.

Augmentation policy (Experiment_Design.md §6): random-resized-crop, color jitter, mild
rotation are "safe, standard" transforms of the provided image. Horizontal flip is
deliberately *not* included by default -- it can invert real geolocation cues (driving
side of the road, signage orientation), so Experiment_Design.md flags it as a judgment
call to test explicitly, not something to assume is free signal. Mixup/cutmix-style
blending is out of scope entirely (pending the organizer's synthetic-data answer).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.geo.countries import COUNTRY_TO_IDX
from src.geo.spherical import (
    latlon_to_local_offset_km_np,
    latlon_to_unit_vector_np,
    unit_vector_to_latlon_np,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO_ROOT / "geo_dataset" / "train"
LABELS_CSV = REPO_ROOT / "geo_dataset" / "train_labels.csv"
FOLDS_JSON = REPO_ROOT / "outputs" / "protocol" / "folds" / "folds.json"
GEO_CELLS_DIR = REPO_ROOT / "outputs" / "protocol" / "geo_cells"
GEO_CELLS_MULTI_DIR = REPO_ROOT / "outputs" / "protocol" / "geo_cells_multi"
GEO_CELLS_REGION_DIR = REPO_ROOT / "outputs" / "protocol" / "geo_cells_region"

# MobileViT (cvnets recipe, via timm) expects [0, 1]-scaled RGB, no ImageNet mean/std
# normalization (verified via timm.data.resolve_data_config: mean=(0,0,0), std=(1,1,1)) --
# matching this exactly matters for the pretrained weights to be used correctly. This is the
# *default* used by every caller that doesn't explicitly ask for a different resolution
# (T2.1's direct-regression pipeline). The hierarchical classifier experiments explicitly
# pass a higher `input_size` (384px) to `build_transform`/`GeoDataset` -- see
# `src/train_hierarchical.py` -- since the dataset ships native 512x512 images and 256px may
# be discarding fine cues (signage text, plate style) that matter for country/cell precision.
INPUT_SIZE = 256


def load_labelled_df() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    fold_map = json.loads(FOLDS_JSON.read_text())["filename_to_fold"]
    df["fold"] = df["filename"].map(fold_map)
    if df["fold"].isna().any():
        raise ValueError("Some rows have no fold assignment -- re-run make_folds.py.")
    df["fold"] = df["fold"].astype(int)
    return df


def load_fold_cells(val_fold: int) -> dict:
    """Load the fixed per-fold, per-country geo-cell partition
    (`src/protocol/geo_cells.py`) -- each cell's spherical-mean centroid + exact country,
    and the full filename -> cell_id assignment (train and val; the tree itself was built
    from train only, see that module's docstring)."""
    path = GEO_CELLS_DIR / f"fold{val_fold}_cells.json"
    data = json.loads(path.read_text())
    n_cells = data["meta"]["n_cells"]
    centroids = np.zeros((n_cells, 2), dtype=np.float64)  # [cell_id] -> (lat, lng)
    cell_to_country = np.zeros(n_cells, dtype=np.int64)  # [cell_id] -> country index
    for cid_str, cell in data["cells"].items():
        cid = int(cid_str)
        centroids[cid] = [cell["centroid_lat"], cell["centroid_lng"]]
        cell_to_country[cid] = COUNTRY_TO_IDX[cell["country"]]
    assignments = {fname: info["cell_id"] for fname, info in data["assignments"].items()}
    return {
        "n_cells": n_cells,
        "centroids": centroids,
        "cell_to_country": cell_to_country,
        "assignments": assignments,
    }


def load_fold_cells_multi(val_fold: int, n_partitions: int) -> dict:
    """T3.4: same as `load_fold_cells`, but for the K independent partitions built by
    `src/protocol/geo_cells_multi.py` (`outputs/protocol/geo_cells_multi/fold{f}_p{k}_cells.json`).
    Returns per-partition lists (index k = partition k) of n_cells/centroids/cell_to_country/
    assignments -- same shape as `load_fold_cells`'s single-partition dict, just K of them."""
    n_cells_list, centroids_list, cell_to_country_list, assignments_list = [], [], [], []
    for k in range(n_partitions):
        path = GEO_CELLS_MULTI_DIR / f"fold{val_fold}_p{k}_cells.json"
        data = json.loads(path.read_text())
        n_cells = data["meta"]["n_cells"]
        centroids = np.zeros((n_cells, 2), dtype=np.float64)
        cell_to_country = np.zeros(n_cells, dtype=np.int64)
        for cid_str, cell in data["cells"].items():
            cid = int(cid_str)
            centroids[cid] = [cell["centroid_lat"], cell["centroid_lng"]]
            cell_to_country[cid] = COUNTRY_TO_IDX[cell["country"]]
        assignments = {fname: info["cell_id"] for fname, info in data["assignments"].items()}
        n_cells_list.append(n_cells)
        centroids_list.append(centroids)
        cell_to_country_list.append(cell_to_country)
        assignments_list.append(assignments)
    return {
        "n_partitions": n_partitions,
        "n_cells": n_cells_list,
        "centroids": centroids_list,
        "cell_to_country": cell_to_country_list,
        "assignments": assignments_list,
    }


def load_fold_region(val_fold: int) -> dict:
    """T4.1d: same shape as `load_fold_cells`, but for the single coarse per-country
    "region" partition built by `src/protocol/geo_cells_region.py`
    (`outputs/protocol/geo_cells_region/fold{f}_cells.json`). Kept SEPARATE from
    `load_fold_cells_multi`'s K fine partitions -- deliberately not folded into that list,
    because a region's centroid is a poor per-image locator (it spans a large chunk of a
    country) and must NOT feed the offset-regression target the way the fine partitions'
    blended centroid does (see `GeoDataset`'s docstring and
    `Experiment_Design.md` Tier 4 T4.1d). The region head only contributes an additional
    conditioning term to retrieval/hierarchy SCORING, not to offset supervision."""
    path = GEO_CELLS_REGION_DIR / f"fold{val_fold}_cells.json"
    data = json.loads(path.read_text())
    n_regions = data["meta"]["n_cells"]
    centroids = np.zeros((n_regions, 2), dtype=np.float64)
    region_to_country = np.zeros(n_regions, dtype=np.int64)
    for cid_str, cell in data["cells"].items():
        cid = int(cid_str)
        centroids[cid] = [cell["centroid_lat"], cell["centroid_lng"]]
        region_to_country[cid] = COUNTRY_TO_IDX[cell["country"]]
    assignments = {fname: info["cell_id"] for fname, info in data["assignments"].items()}
    return {
        "n_regions": n_regions, "centroids": centroids,
        "region_to_country": region_to_country, "assignments": assignments,
    }


def blended_centroid_np(centroids: np.ndarray) -> tuple[float, float]:
    """T3.4: spherical mean of K (lat, lng) centroids (shape (K, 2)) -- the "combined
    estimate" from K independent partitions' cell picks, consistent with this project's
    existing spherical-mean convention (B1 baseline, geo_cells.py's per-cell centroids)
    rather than a naive degree average."""
    vecs = latlon_to_unit_vector_np(centroids[:, 0], centroids[:, 1])
    mean_vec = vecs.mean(axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)
    lat_c, lng_c = unit_vector_to_latlon_np(mean_vec[None, :])
    return float(lat_c[0]), float(lng_c[0])


def build_transform(train: bool, input_size: int = INPUT_SIZE) -> transforms.Compose:
    if train:
        # Widened from the original (scale=(0.7,1.0), no erasing) once T2.1's first real runs
        # showed overfitting (train median km well below val, Experiment_Design.md results
        # log) rather than the underfitting the original policy was tuned for.
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    input_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
                transforms.RandomRotation(10, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),  # scales to [0, 1], matches MobileViT's expected input
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(
                int(input_size / 0.9), interpolation=transforms.InterpolationMode.BICUBIC
            ),  # crop_pct=0.9, per MobileViT-XS's timm data config
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ]
    )


class GeoDataset(Dataset):
    """`df` rows already filtered to the desired fold(s) by the caller (see
    `make_train_val_datasets`) -- keeps this class a plain image+target dataset rather than
    also owning fold-selection logic.

    `cell_info` (optional, from `load_fold_cells`) additionally hands back each item's
    geo-cell id and its local offset from that cell's centroid, in km (see
    `src/geo/spherical.py`'s local-offset functions) -- `None` (default) keeps this class
    exactly as T2.1 uses it, no cell fields in the returned dict. `country_idx` is always
    included (cheap, just a label lookup) since both the direct-regression and hierarchical
    pipelines can ignore it freely.

    `multi_cell_info` (optional, from `load_fold_cells_multi`, T3.4): the multi-partition
    counterpart of `cell_info` -- mutually exclusive with it (only one experiment's cell
    scheme is ever active per dataset instance). Hands back a `cell_ids` vector (one id per
    partition) and an `offset_km` computed relative to the BLENDED (spherical-mean) centroid
    across all K partitions' true cells, not any single partition's centroid -- see
    `blended_centroid_np`.

    `region_info` (optional, from `load_fold_region`, T4.1d): ADDITIVE, not mutually
    exclusive with the others -- can be combined with `multi_cell_info` to also hand back a
    `region_id` scalar. Deliberately NOT folded into `multi_cell_info`'s partition list or
    its offset-target blend (see `load_fold_region`'s docstring for why).
    """

    def __init__(
        self, df: pd.DataFrame, train: bool, input_size: int = INPUT_SIZE,
        cell_info: dict | None = None, multi_cell_info: dict | None = None,
        region_info: dict | None = None,
    ):
        assert cell_info is None or multi_cell_info is None, (
            "cell_info and multi_cell_info are mutually exclusive"
        )
        self.df = df.reset_index(drop=True)
        self.transform = build_transform(train, input_size=input_size)
        self.cell_info = cell_info
        self.multi_cell_info = multi_cell_info
        self.region_info = region_info

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(IMAGES_DIR / row["filename"]).convert("RGB")
        image = self.transform(image)

        target_vec = latlon_to_unit_vector_np(np.array([row["lat"]]), np.array([row["lng"]]))[0]
        target_vec = torch.as_tensor(target_vec, dtype=torch.float32)

        item = {
            "image": image,
            "target_vec": target_vec,
            "lat": torch.tensor(row["lat"], dtype=torch.float64),
            "lng": torch.tensor(row["lng"], dtype=torch.float64),
            "country": row["country"],
            "country_idx": torch.tensor(COUNTRY_TO_IDX[row["country"]], dtype=torch.long),
            "filename": row["filename"],
        }

        if self.cell_info is not None:
            cell_id = self.cell_info["assignments"][row["filename"]]
            centroid_lat, centroid_lng = self.cell_info["centroids"][cell_id]
            offset_km = latlon_to_local_offset_km_np(
                np.array([row["lat"]]),
                np.array([row["lng"]]),
                np.array([centroid_lat]),
                np.array([centroid_lng]),
            )[0]
            item["cell_id"] = torch.tensor(cell_id, dtype=torch.long)
            item["offset_km"] = torch.as_tensor(offset_km, dtype=torch.float32)

        if self.multi_cell_info is not None:
            K = self.multi_cell_info["n_partitions"]
            cell_ids = [
                self.multi_cell_info["assignments"][k][row["filename"]] for k in range(K)
            ]
            centroids_k = np.array(
                [self.multi_cell_info["centroids"][k][cell_ids[k]] for k in range(K)]
            )  # (K, 2)
            blend_lat, blend_lng = blended_centroid_np(centroids_k)
            offset_km = latlon_to_local_offset_km_np(
                np.array([row["lat"]]), np.array([row["lng"]]),
                np.array([blend_lat]), np.array([blend_lng]),
            )[0]
            item["cell_ids"] = torch.as_tensor(cell_ids, dtype=torch.long)
            item["offset_km"] = torch.as_tensor(offset_km, dtype=torch.float32)

        if self.region_info is not None:
            region_id = self.region_info["assignments"][row["filename"]]
            item["region_id"] = torch.tensor(region_id, dtype=torch.long)

        return item


def make_train_val_datasets(
    val_fold: int, input_size: int = INPUT_SIZE, use_cells: bool = False
) -> tuple[GeoDataset, GeoDataset]:
    """Standard CV split for one fold: `val_fold` held out for validation, the rest for
    training -- this is what both the quick 1-2 fold screen and the full 5-fold run call,
    just with a different set of `val_fold` values.

    `use_cells=True` loads that fold's geo-cell partition once and hands the same
    `cell_info` to both the train and val datasets -- correct because the trees themselves
    were already built from train only (`geo_cells.py`); reusing them for val here is just
    an assignment lookup, not a second construction from val data.
    """
    df = load_labelled_df()
    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]
    cell_info = load_fold_cells(val_fold) if use_cells else None
    return (
        GeoDataset(train_df, train=True, input_size=input_size, cell_info=cell_info),
        GeoDataset(val_df, train=False, input_size=input_size, cell_info=cell_info),
    )


def make_dataloaders(
    val_fold: int, batch_size: int = 128, num_workers: int = 8,
    input_size: int = INPUT_SIZE, use_cells: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_ds, val_ds = make_train_val_datasets(val_fold, input_size=input_size, use_cells=use_cells)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def make_train_val_datasets_multi(
    val_fold: int, n_partitions: int, input_size: int = INPUT_SIZE
) -> tuple[GeoDataset, GeoDataset]:
    """T3.4 counterpart of `make_train_val_datasets` -- always loads the K-partition cell
    scheme (`use_cells`-style toggle doesn't apply, this function only exists for T3.4)."""
    df = load_labelled_df()
    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]
    multi_cell_info = load_fold_cells_multi(val_fold, n_partitions)
    return (
        GeoDataset(train_df, train=True, input_size=input_size, multi_cell_info=multi_cell_info),
        GeoDataset(val_df, train=False, input_size=input_size, multi_cell_info=multi_cell_info),
    )


def make_dataloaders_multi(
    val_fold: int, n_partitions: int, batch_size: int = 128, num_workers: int = 8,
    input_size: int = INPUT_SIZE,
) -> tuple[DataLoader, DataLoader]:
    train_ds, val_ds = make_train_val_datasets_multi(
        val_fold, n_partitions, input_size=input_size
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
