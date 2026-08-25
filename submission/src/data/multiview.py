"""Global/local views and geographically balanced batches for T4.1b."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler
from torchvision import transforms
from torchvision.transforms import functional as TF


class GlobalLocalTransform:
    """Return one full-scene view and four corner-detail views.

    All crops come from the provided image only.  The evaluation path is deterministic;
    training adds only the challenge-legal photometric and mild rotation augmentations.
    """

    def __init__(
        self,
        input_size: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        crop_pct: float,
        train: bool,
        local_fraction: float = 0.64,
    ) -> None:
        self.input_size = input_size
        self.train = train
        self.local_fraction = local_fraction
        normalize = transforms.Normalize(mean=mean, std=std)
        if train:
            self.global_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    input_size, scale=(0.65, 1.0), ratio=(0.85, 1.15),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
                transforms.RandomRotation(
                    5, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(), normalize,
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
            ])
            self.local_transform = transforms.Compose([
                transforms.Resize(
                    (input_size, input_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ColorJitter(0.15, 0.15, 0.15, 0.015),
                transforms.RandomRotation(
                    3, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(), normalize,
                transforms.RandomErasing(p=0.05, scale=(0.02, 0.06)),
            ])
        else:
            self.global_transform = transforms.Compose([
                transforms.Resize(
                    round(input_size / crop_pct),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(input_size), transforms.ToTensor(), normalize,
            ])
            self.local_transform = transforms.Compose([
                transforms.Resize(
                    (input_size, input_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(), normalize,
            ])

    def __call__(self, image):
        width, height = image.size
        side = max(1, round(min(width, height) * self.local_fraction))
        max_left, max_top = width - side, height - side
        corners = [(0, 0), (max_left, 0), (0, max_top), (max_left, max_top)]
        if self.train:
            # Small inward/outward jitter preserves each quadrant's meaning.
            jitter = max(1, round(side * 0.06))
            corners = [
                (
                    int(np.clip(left + np.random.randint(-jitter, jitter + 1), 0, max_left)),
                    int(np.clip(top + np.random.randint(-jitter, jitter + 1), 0, max_top)),
                )
                for left, top in corners
            ]
        views = [self.global_transform(image)]
        for left, top in corners:
            crop = TF.crop(image, top, left, side, side)
            views.append(self.local_transform(crop))
        return torch.stack(views)


class GeoBalancedBatchSampler(Sampler[list[int]]):
    """Sample several examples from each geographic cell in every batch."""

    def __init__(
        self,
        filenames: list[str],
        assignments: dict[str, int],
        batch_size: int,
        samples_per_cell: int = 4,
        seed: int = 1337,
    ) -> None:
        if batch_size % samples_per_cell:
            raise ValueError("batch_size must be divisible by samples_per_cell")
        groups: dict[int, list[int]] = defaultdict(list)
        for index, filename in enumerate(filenames):
            groups[assignments[filename]].append(index)
        self.groups = list(groups.values())
        self.batch_size = batch_size
        self.samples_per_cell = samples_per_cell
        self.cells_per_batch = batch_size // samples_per_cell
        self.n_batches = len(filenames) // batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.n_batches):
            chosen = rng.choice(
                len(self.groups), size=self.cells_per_batch,
                replace=len(self.groups) < self.cells_per_batch,
            )
            batch: list[int] = []
            for group_id in chosen:
                group = self.groups[int(group_id)]
                sample = rng.choice(
                    group, size=self.samples_per_cell,
                    replace=len(group) < self.samples_per_cell,
                )
                batch.extend(int(index) for index in sample)
            rng.shuffle(batch)
            yield batch
