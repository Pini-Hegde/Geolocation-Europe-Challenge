"""Shared-backbone global/local retrieval model for T4.1b."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.retrieval_geo_classifier import RetrievalGeoClassifier


class MultiViewRetrievalGeoClassifier(RetrievalGeoClassifier):
    """Run all views through one backbone and fuse them without extra parameters."""

    def forward(self, images: torch.Tensor):
        if images.ndim != 5:
            raise ValueError(f"expected [batch, views, channels, height, width], got {images.shape}")
        batch, n_views, channels, height, width = images.shape
        view_features = self.backbone(
            images.reshape(batch * n_views, channels, height, width)
        ).reshape(batch, n_views, -1)
        # Equal global/local contribution prevents four detail crops overwhelming context.
        local_mean = view_features[:, 1:].mean(dim=1)
        fused = self.dropout(0.5 * (view_features[:, 0] + local_mean))
        country_logits = self.country_head(fused)
        cell_logits = [head(fused) for head in self.cls_heads]
        offset = self.offset_head(fused)
        view_embeddings = F.normalize(
            self.embedding_head(self.dropout(view_features)), dim=-1, eps=1e-8
        )
        embedding = F.normalize(
            0.5 * (view_embeddings[:, 0] + view_embeddings[:, 1:].mean(dim=1)),
            dim=-1, eps=1e-8,
        )
        return cell_logits, offset, country_logits, embedding, view_embeddings
