"""T4.1d: T4.1c's shared global/local multi-view retrieval encoder PLUS one additional
coarse country->region head, for retrieval-score conditioning.

Motivation, and why this is a NEW file rather than an edit to
`multiview_retrieval_geo_classifier.py`: see `src/protocol/geo_cells_region.py`'s module
docstring and this project's standing convention (T3.4's `multi_partition_geo_classifier.py`,
the reverted T3.3/T3.6 attempt) of never modifying a working baseline file in place.

The region head is intentionally NOT part of `cls_heads` (the K fine cell partitions) --
its logits are not used for the offset-regression target or the spherical blended centroid
the offset head predicts relative to (a region spans a large chunk of a country and would be
a poor per-image locator; see `load_fold_region`'s docstring in `src/data/dataset.py`). It
only contributes an extra term to retrieval/hierarchy SCORING at inference and its own masked
cross-entropy loss at training -- see `src/train_region_retrieval_geo.py`.

Forward is a full reimplementation of `MultiViewRetrievalGeoClassifier.forward`, not a call to
`super().forward()` -- that method doesn't expose the intermediate fused feature the region
head needs, and recomputing it separately would run the backbone twice per batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geo.countries import N_COUNTRIES
from src.models.retrieval_geo_classifier import RetrievalGeoClassifier


class RegionMultiViewRetrievalGeoClassifier(RetrievalGeoClassifier):
    def __init__(
        self,
        backbone_name: str,
        n_cells_per_partition: list[int],
        cell_to_country_per_partition: list,
        n_regions: int,
        region_to_country,
        embedding_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            backbone_name, n_cells_per_partition, cell_to_country_per_partition,
            embedding_dim=embedding_dim, pretrained=pretrained, dropout=dropout,
        )
        feature_dim = self.backbone.num_features
        self.n_regions = n_regions
        self.region_head = nn.Linear(feature_dim, n_regions)

        mapping_t = torch.as_tensor(region_to_country, dtype=torch.long)
        if mapping_t.shape != (n_regions,):
            raise ValueError(f"expected region country map ({n_regions},), got {mapping_t.shape}")
        mask = torch.zeros(N_COUNTRIES, n_regions, dtype=torch.bool)
        mask[mapping_t, torch.arange(n_regions)] = True
        self.register_buffer("country_region_mask", mask, persistent=False)

    def forward(self, images: torch.Tensor):
        if images.ndim != 5:
            raise ValueError(f"expected [batch, views, channels, height, width], got {images.shape}")
        batch, n_views, channels, height, width = images.shape
        view_features = self.backbone(
            images.reshape(batch * n_views, channels, height, width)
        ).reshape(batch, n_views, -1)
        local_mean = view_features[:, 1:].mean(dim=1)
        fused = self.dropout(0.5 * (view_features[:, 0] + local_mean))
        country_logits = self.country_head(fused)
        cell_logits = [head(fused) for head in self.cls_heads]
        region_logits = self.region_head(fused)
        offset = self.offset_head(fused)
        view_embeddings = F.normalize(
            self.embedding_head(self.dropout(view_features)), dim=-1, eps=1e-8
        )
        embedding = F.normalize(
            0.5 * (view_embeddings[:, 0] + view_embeddings[:, 1:].mean(dim=1)),
            dim=-1, eps=1e-8,
        )
        return cell_logits, offset, country_logits, region_logits, embedding, view_embeddings
