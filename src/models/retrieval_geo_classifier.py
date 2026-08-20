"""Parameter-budgeted classifier plus geographic retrieval embedding.

This model is separate from the T3.4 baseline. Cell logits remain unmasked so inference
can combine country and conditional cell probabilities without an irreversible country
argmax inside the network.
"""

from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geo.countries import N_COUNTRIES


class RetrievalGeoClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        n_cells_per_partition: list[int],
        cell_to_country_per_partition: list[np.ndarray],
        embedding_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(n_cells_per_partition) != len(cell_to_country_per_partition):
            raise ValueError("cell counts and country maps must have equal length")

        self.backbone_name = backbone_name
        self.n_partitions = len(n_cells_per_partition)
        self.n_cells_per_partition = list(n_cells_per_partition)
        self.embedding_dim = embedding_dim
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.country_head = nn.Linear(feature_dim, N_COUNTRIES)
        self.cls_heads = nn.ModuleList(
            nn.Linear(feature_dim, n_cells) for n_cells in n_cells_per_partition
        )
        self.offset_head = nn.Linear(feature_dim, 2)
        self.embedding_head = nn.Linear(feature_dim, embedding_dim)

        for k, (n_cells, mapping) in enumerate(
            zip(n_cells_per_partition, cell_to_country_per_partition)
        ):
            mapping_t = torch.as_tensor(mapping, dtype=torch.long)
            if mapping_t.shape != (n_cells,):
                raise ValueError(
                    f"partition {k}: expected country map ({n_cells},), got {mapping_t.shape}"
                )
            mask = torch.zeros(N_COUNTRIES, n_cells, dtype=torch.bool)
            mask[mapping_t, torch.arange(n_cells)] = True
            self.register_buffer(f"country_cell_mask_{k}", mask, persistent=False)

    def country_cell_mask(self, partition: int) -> torch.Tensor:
        return getattr(self, f"country_cell_mask_{partition}")

    def forward(self, images: torch.Tensor):
        features = self.dropout(self.backbone(images))
        country_logits = self.country_head(features)
        cell_logits = [head(features) for head in self.cls_heads]
        offset = self.offset_head(features)
        embedding = F.normalize(self.embedding_head(features), dim=-1, eps=1e-8)
        return cell_logits, offset, country_logits, embedding
