"""
Hierarchical country -> within-country-cell classifier + offset regression.

Design rationale (Experiment_Design.md Tier 3 discussion, 2026-08-19): an earlier attempt
used a single flat classifier over a global (country-agnostic) cell partition, with a
*soft* country bias added to the cell logits afterward (a learned scalar gate). That only
nudged accuracy, because the bias arrives after the hard decision is already being made
jointly across all 12 countries' cells at once -- the fine classifier still has to learn to
separate a Germany-shaped cell from a Poland-shaped cell using the same shared weights as
every other pair.

This version makes the hierarchy *structural* instead of a hint:
  1. `country_head` predicts the country (12-way).
  2. `cls_head` produces logits over ALL cells globally, but before they're used for loss or
     argmax, they are **hard-masked** to only the cells belonging to one country (via a fixed
     boolean `country_cell_mask` built from `src/protocol/geo_cells.py`'s country-respecting
     partition, where every cell belongs to exactly one country by construction). The masked-
     out logits get a large negative value, not -inf, to stay stable under AMP/fp16.
  3. `offset_head` regresses the local offset from the (masked-argmax) cell's centroid.

Which country's mask to apply differs between training and inference:
  - **Training**: mask by the TRUE country (teacher forcing, via `true_country_idx`). This
    keeps the cell head's supervision clean and unambiguous -- it only ever has to learn to
    rank cells *within* the correct country, never has to also discover which country is
    right from a masked, mostly -inf signal.
  - **Inference/eval** (`true_country_idx=None`): mask by the country head's OWN prediction
    (argmax). This is the only information available at prediction time, and it's exactly
    what makes cross-country cell errors structurally impossible *given a correct country
    prediction* -- if the country head is right, the cell head cannot pick a cell from a
    different country, unlike the earlier soft-gate version.

`n_cells` and `cell_to_country` are fold-specific (each fold's cell partition is built from
that fold's own training split) -- this model is instantiated fresh per fold.
"""

from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn

from src.geo.countries import N_COUNTRIES

# Large-but-finite mask value, not -inf: keeps softmax/cross-entropy well-defined and
# gradient-stable under torch.autocast (fp16 -inf can produce NaNs on some ops/hardware).
MASK_VALUE = -1e4


class HierarchicalGeoClassifierRegressor(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        n_cells: int,
        cell_to_country: np.ndarray,
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert cell_to_country.shape == (n_cells,), (
            f"cell_to_country must have shape ({n_cells},), got {cell_to_country.shape}"
        )
        self.backbone_name = backbone_name
        self.n_cells = n_cells
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(p=dropout)
        self.country_head = nn.Linear(self.backbone.num_features, N_COUNTRIES)
        self.cls_head = nn.Linear(self.backbone.num_features, n_cells)
        self.offset_head = nn.Linear(self.backbone.num_features, 2)

        # Fixed, non-learned (N_COUNTRIES, n_cells) boolean mask: country_cell_mask[c, i] is
        # True iff cell i belongs to country c. Built once from geo_cells.py's exact
        # (not majority-estimated) per-cell country field. Registered as a buffer so it
        # moves with the model across .to(device) calls, not counted as a trained parameter.
        cell_to_country_t = torch.as_tensor(cell_to_country, dtype=torch.long)
        country_cell_mask = torch.zeros(N_COUNTRIES, n_cells, dtype=torch.bool)
        country_cell_mask[cell_to_country_t, torch.arange(n_cells)] = True
        self.register_buffer("country_cell_mask", country_cell_mask, persistent=False)

    def forward(self, images, true_country_idx: "torch.Tensor | None" = None):
        features = self.backbone(images)
        features = self.dropout(features)

        country_logits = self.country_head(features)
        cell_logits_raw = self.cls_head(features)
        offset = self.offset_head(features)

        if true_country_idx is not None:
            mask_idx = true_country_idx  # training: teacher forcing
        else:
            mask_idx = country_logits.argmax(dim=-1)  # inference/eval: self-prediction

        allowed = self.country_cell_mask[mask_idx]  # (B, n_cells) bool
        cell_logits = torch.where(
            allowed, cell_logits_raw, torch.full_like(cell_logits_raw, MASK_VALUE)
        )

        return cell_logits, offset, country_logits
