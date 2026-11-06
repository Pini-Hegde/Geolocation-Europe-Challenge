"""
Backbone + spherical-vector regression head (Experiment_Design.md Part 2, Tier 2).

A thin wrapper around a `timm` backbone (feature extractor, `num_classes=0`) followed by a
linear head producing 3 raw outputs -- the unnormalized (x, y, z) vector. Deliberately kept
generic over the backbone name (not hardcoded to MobileViT-XS) so T2.2-T2.4's comparison
backbones (MobileNetV3-Small, GhostNet, ShuffleNetV2) reuse this same class instead of each
needing their own model file.

The head output is *not* normalized to unit length inside the model -- `angular_loss`
normalizes internally during training, and `unit_vector_to_latlon_torch` normalizes at
inference -- so this class stays a plain nn.Module with no extra logic to keep param-count
verification (`src/protocol/param_check.py`) simple and exact.
"""

from __future__ import annotations

import timm
import torch.nn as nn


class GeoBackboneRegressor(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        # Dropout right before the head -- a cheap, targeted regularizer for overfitting seen
        # in fine-tuning (Experiment_Design.md results log, T2.1): train median km well below
        # val, on a pretrained backbone whose own normalization layers are already fixed by
        # its architecture (nothing to add there). p=0.0 is a no-op (nn.Dropout(0) is identity),
        # so this stays backward compatible with runs that don't pass --dropout.
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(self.backbone.num_features, 3)

    def forward(self, images):
        features = self.backbone(images)
        features = self.dropout(features)
        return self.head(features)
