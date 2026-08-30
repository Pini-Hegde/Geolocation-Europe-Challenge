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
        cell_proj_dim: int | None = None,
        use_location_encoder: bool = False,
        country_head_hidden: int = 0,
        region_to_country: "np.ndarray | None" = None,
    ) -> None:
        super().__init__()
        if len(n_cells_per_partition) != len(cell_to_country_per_partition):
            raise ValueError("cell counts and country maps must have equal length")

        self.backbone_name = backbone_name
        self.n_partitions = len(n_cells_per_partition)
        self.n_cells_per_partition = list(n_cells_per_partition)
        self.embedding_dim = embedding_dim
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        # timm's `num_features` is unreliable for some backbones (mobilenetv4, hgnetv2 have a
        # conv_head that expands the channel dim after the trunk and isn't reflected when
        # num_classes=0). Probe the real pooled output width with a dummy forward.
        self.backbone.eval()
        with torch.no_grad():
            feature_dim = int(self.backbone(torch.zeros(2, 3, 224, 224)).shape[-1])
        self.backbone.train()
        self.dropout = nn.Dropout(dropout)
        # T6/E3g: France/Italy/Spain country accuracy (~0.6) is the fold-0 bottleneck. A small
        # MLP head gives the 12-way continental-Europe separation more capacity than a single
        # Linear. country_head_hidden=0 keeps the original Linear head.
        if country_head_hidden > 0:
            self.country_head = nn.Sequential(
                nn.Linear(feature_dim, country_head_hidden), nn.GELU(),
                nn.Linear(country_head_hidden, N_COUNTRIES),
            )
        else:
            self.country_head = nn.Linear(feature_dim, N_COUNTRIES)
        # T6/E3+: a shared low-rank projection before the K per-partition cell classifiers,
        # so the parameter cost of many fine cells is K * cell_proj_dim * n_cells instead of
        # K * feature_dim(=1280) * n_cells -- lets K>=3 partitions of ~200-400 cells each fit
        # under the 5M cap. cell_proj_dim=None keeps the original direct feature->cells heads.
        self.cell_proj_dim = cell_proj_dim
        if cell_proj_dim is not None:
            self.cell_trunk = nn.Sequential(
                nn.Linear(feature_dim, cell_proj_dim), nn.GELU()
            )
            head_in = cell_proj_dim
        else:
            self.cell_trunk = nn.Identity()
            head_in = feature_dim
        self.cls_heads = nn.ModuleList(
            nn.Linear(head_in, n_cells) for n_cells in n_cells_per_partition
        )
        self.offset_head = nn.Linear(feature_dim, 2)
        self.embedding_head = nn.Linear(feature_dim, embedding_dim)

        # T6/E8: an intermediate country->region head (Müller-Budack / GeoGuessNet hierarchy).
        # ~2-4 coarse regions per country -- an easy classification target (high accuracy)
        # that bounds the error when the fine cell is wrong but the country is right (the
        # regime of the accurate countries, whose within-country cell errors set the median).
        self.n_regions = 0
        if region_to_country is not None:
            rt = torch.as_tensor(region_to_country, dtype=torch.long)
            self.n_regions = int(rt.numel())
            self.region_head = nn.Linear(feature_dim, self.n_regions)
            rmask = torch.zeros(N_COUNTRIES, self.n_regions, dtype=torch.bool)
            rmask[rt, torch.arange(self.n_regions)] = True
            self.register_buffer("country_region_mask", rmask, persistent=False)

        # T6/E4: GeoCLIP-style continuous location encoder (Cepeda et al. 2023). Maps a
        # (lat, lng) to the same embedding space as the image, so the decoder can score the
        # image against *any* coordinate, not only the labelled training points -- fills the
        # gaps in sparsely-sampled large countries.
        self.use_location_encoder = use_location_encoder
        if use_location_encoder:
            from src.models.location_encoder import LocationEncoder

            # kept small (~90k params) to fit alongside K>=3 fine-cell heads under the 5M cap
            self.location_encoder = LocationEncoder(
                embedding_dim=embedding_dim, n_scales=2, fourier_features=48, hidden_dim=96
            )

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

    def encode_location(self, lat_deg: torch.Tensor, lng_deg: torch.Tensor) -> torch.Tensor:
        return self.location_encoder(lat_deg, lng_deg)

    def forward(self, images: torch.Tensor):
        features = self.dropout(self.backbone(images))
        country_logits = self.country_head(features)
        cell_feat = self.cell_trunk(features)
        cell_logits = [head(cell_feat) for head in self.cls_heads]
        offset = self.offset_head(features)
        embedding = F.normalize(self.embedding_head(features), dim=-1, eps=1e-8)
        return cell_logits, offset, country_logits, embedding
