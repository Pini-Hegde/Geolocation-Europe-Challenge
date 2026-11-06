"""
Multi-partition hierarchical country -> within-country-cell classifier + offset regression
(T3.4, Experiment_Design.md Tier 3). A separate model class from
`hierarchical_geo_classifier.py` (T3.1) on purpose -- T3.1 is the working baseline every
other candidate is measured against, so it stays untouched rather than gaining a mode switch
(the T3.3/T3.6 attempt left flags on that file before being reverted; this project's
convention now is a new file per architecture change, not a flag on the baseline).

Design (literature.md §1, CPlaNet): instead of one geo-cell partition (T3.1: 96 cells,
~100 images/cell, cell-classification accuracy capped at 41.8%), classify against K
independent partitions of the SAME points (`src/protocol/geo_cells_multi.py`) at the SAME
per-partition granularity as T3.1 (so per-partition difficulty/data-density is unchanged --
only the "multiple independent views, then combine" mechanism is the tested variable). Each
partition gets its own small linear classification head, all still hard-masked by country
exactly as in T3.1 (`country_cell_mask_k`, teacher-forced by true country at training,
self-predicted at eval). At inference, the K heads' independently-argmaxed cell centroids are
combined (spherical mean, `blended_centroid_np`) into one finer effective location estimate
BEFORE the single shared offset head's prediction is added -- two independent coarse
decisions triangulate a finer point than either alone, without needing enough data to
populate a directly-finer joint cell space (CPlaNet's core insight, ported at a much smaller
K given this project's ~11.7k-image scale vs. the paper's web-scale data).

Params: only K small `Linear(feat_dim, n_cells_per_partition)` heads instead of T3.1's one --
cheap (see `src/protocol/param_check.py`'s `check_compliance_multi_partition_model`), not a
new randomly-initialized mechanism competing with the backbone for gradient signal the way
T3.6's attention pool was (see `failed_experiments/T3.3_T3.6_soft_labels_attention_pool/`) --
this is the reasoning behind expecting T3.4 to be lower-risk than T3.6 was.
"""

from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn

from src.geo.countries import N_COUNTRIES

# Same convention as hierarchical_geo_classifier.py -- large-but-finite, not -inf, for AMP
# stability.
MASK_VALUE = -1e4


class MultiPartitionGeoClassifierRegressor(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        n_cells_per_partition: list[int],
        cell_to_country_per_partition: list[np.ndarray],
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_partitions = len(n_cells_per_partition)
        assert self.n_partitions == len(cell_to_country_per_partition), (
            "n_cells_per_partition and cell_to_country_per_partition must have the same length"
        )
        self.backbone_name = backbone_name
        self.n_cells_per_partition = list(n_cells_per_partition)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(p=dropout)
        self.country_head = nn.Linear(self.backbone.num_features, N_COUNTRIES)
        self.cls_heads = nn.ModuleList(
            [nn.Linear(self.backbone.num_features, n) for n in n_cells_per_partition]
        )
        self.offset_head = nn.Linear(self.backbone.num_features, 2)

        # One (N_COUNTRIES, n_cells_k) mask buffer per partition -- same construction as
        # hierarchical_geo_classifier.py's single mask, just K of them.
        for k, (n_cells, cell_to_country) in enumerate(
            zip(n_cells_per_partition, cell_to_country_per_partition)
        ):
            assert cell_to_country.shape == (n_cells,), (
                f"partition {k}: cell_to_country must have shape ({n_cells},), "
                f"got {cell_to_country.shape}"
            )
            cell_to_country_t = torch.as_tensor(cell_to_country, dtype=torch.long)
            mask = torch.zeros(N_COUNTRIES, n_cells, dtype=torch.bool)
            mask[cell_to_country_t, torch.arange(n_cells)] = True
            self.register_buffer(f"country_cell_mask_{k}", mask, persistent=False)

    def forward(self, images, true_country_idx: "torch.Tensor | None" = None):
        features = self.backbone(images)
        features = self.dropout(features)

        country_logits = self.country_head(features)
        offset = self.offset_head(features)

        if true_country_idx is not None:
            mask_idx = true_country_idx  # training: teacher forcing
        else:
            mask_idx = country_logits.argmax(dim=-1)  # inference/eval: self-prediction

        cell_logits_list = []
        for k in range(self.n_partitions):
            raw_k = self.cls_heads[k](features)
            mask_k = getattr(self, f"country_cell_mask_{k}")[mask_idx]  # (B, n_cells_k) bool
            cell_logits_list.append(
                torch.where(mask_k, raw_k, torch.full_like(raw_k, MASK_VALUE))
            )

        return cell_logits_list, offset, country_logits
