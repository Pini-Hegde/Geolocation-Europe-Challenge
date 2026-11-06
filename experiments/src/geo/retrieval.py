"""Soft hierarchical gallery scoring and spherical top-k coordinate refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def conditional_cell_log_probs(
    logits: torch.Tensor, country_cell_mask: torch.Tensor
) -> torch.Tensor:
    """Normalize each country's cells independently to obtain log P(cell | country)."""
    out = torch.full_like(logits, -1e4)
    for country_idx in range(country_cell_mask.shape[0]):
        mask = country_cell_mask[country_idx]
        out[:, mask] = F.log_softmax(logits[:, mask], dim=-1)
    return out


def gallery_scores(
    country_logits: torch.Tensor,
    cell_logits_list: list[torch.Tensor],
    query_embeddings: torch.Tensor,
    country_cell_masks: list[torch.Tensor],
    gallery_country: torch.Tensor,
    gallery_cell_ids: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    retrieval_weight: float,
) -> torch.Tensor:
    """Score every labelled gallery location for every query.

    The hierarchy term is log P(country) plus the mean conditional log-probability of
    that gallery point's cells. The retrieval term adds continuous visual similarity.
    """
    country_lp = F.log_softmax(country_logits.float(), dim=-1)
    scores = country_lp[:, gallery_country]
    for k, logits in enumerate(cell_logits_list):
        cell_lp = conditional_cell_log_probs(logits.float(), country_cell_masks[k])
        scores = scores + cell_lp[:, gallery_cell_ids[:, k]] / len(cell_logits_list)
    if retrieval_weight:
        scores = scores + retrieval_weight * (
            query_embeddings.float() @ gallery_embeddings.float().T
        )
    return scores


def spherical_topk_decode(
    scores: torch.Tensor,
    gallery_unit_vectors: torch.Tensor,
    top_k: int = 5,
    vote_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode with a score-weighted spherical mean of top-k gallery coordinates."""
    k = min(top_k, scores.shape[1])
    top_scores, top_indices = scores.topk(k, dim=-1)
    weights = F.softmax(top_scores / vote_temperature, dim=-1)
    vectors = gallery_unit_vectors[top_indices]
    pred_vec = F.normalize((weights.unsqueeze(-1) * vectors).sum(dim=1), dim=-1, eps=1e-8)
    x, y, z = pred_vec.unbind(dim=-1)
    pred_lat = torch.rad2deg(torch.asin(z.clamp(-1.0, 1.0)))
    pred_lng = torch.rad2deg(torch.atan2(y, x))
    return pred_lat, pred_lng, top_indices
