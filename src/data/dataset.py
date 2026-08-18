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

from src.geo.spherical import latlon_to_unit_vector_np

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO_ROOT / "geo_dataset" / "train"
LABELS_CSV = REPO_ROOT / "geo_dataset" / "train_labels.csv"
FOLDS_JSON = REPO_ROOT / "outputs" / "protocol" / "folds" / "folds.json"

# MobileViT (cvnets recipe, via timm) expects [0, 1]-scaled RGB, no ImageNet mean/std
# normalization (verified via timm.data.resolve_data_config: mean=(0,0,0), std=(1,1,1)) --
# matching this exactly matters for the pretrained weights to be used correctly.
INPUT_SIZE = 256


def load_labelled_df() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    fold_map = json.loads(FOLDS_JSON.read_text())["filename_to_fold"]
    df["fold"] = df["filename"].map(fold_map)
    if df["fold"].isna().any():
        raise ValueError("Some rows have no fold assignment -- re-run make_folds.py.")
    df["fold"] = df["fold"].astype(int)
    return df


def build_transform(train: bool) -> transforms.Compose:
    if train:
        # Widened from the original (scale=(0.7,1.0), no erasing) once T2.1's first real runs
        # showed overfitting (train median km well below val, Experiment_Design.md results
        # log) rather than the underfitting the original policy was tuned for.
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    INPUT_SIZE, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
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
                int(INPUT_SIZE / 0.9), interpolation=transforms.InterpolationMode.BICUBIC
            ),  # crop_pct=0.9, per MobileViT-XS's timm data config
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
        ]
    )


class GeoDataset(Dataset):
    """`df` rows already filtered to the desired fold(s) by the caller (see
    `make_train_val_datasets`) -- keeps this class a plain image+target dataset rather than
    also owning fold-selection logic."""

    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.transform = build_transform(train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(IMAGES_DIR / row["filename"]).convert("RGB")
        image = self.transform(image)

        target_vec = latlon_to_unit_vector_np(np.array([row["lat"]]), np.array([row["lng"]]))[0]
        target_vec = torch.as_tensor(target_vec, dtype=torch.float32)

        return {
            "image": image,
            "target_vec": target_vec,
            "lat": torch.tensor(row["lat"], dtype=torch.float64),
            "lng": torch.tensor(row["lng"], dtype=torch.float64),
            "country": row["country"],
            "filename": row["filename"],
        }


def make_train_val_datasets(val_fold: int) -> tuple[GeoDataset, GeoDataset]:
    """Standard CV split for one fold: `val_fold` held out for validation, the rest for
    training -- this is what both the quick 1-2 fold screen and the full 5-fold run call,
    just with a different set of `val_fold` values."""
    df = load_labelled_df()
    train_df = df[df["fold"] != val_fold]
    val_df = df[df["fold"] == val_fold]
    return GeoDataset(train_df, train=True), GeoDataset(val_df, train=False)


def make_dataloaders(
    val_fold: int, batch_size: int = 128, num_workers: int = 8
) -> tuple[DataLoader, DataLoader]:
    train_ds, val_ds = make_train_val_datasets(val_fold)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
