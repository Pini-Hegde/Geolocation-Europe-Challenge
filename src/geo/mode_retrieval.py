"""Mode-aware geographic decoding for retrieval candidates."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spherical_mode_decode(
    scores: torch.Tensor,
    gallery_unit_vectors: torch.Tensor,
    top_m: int = 50,
    radius_km: float = 100.0,
    vote_temperature: float = 1.0,
    kernel: str = "gaussian",
    geometric_median_iters: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the strongest coherent geographic mode before spherical averaging.

    A plain top-k mean can average mutually exclusive hypotheses in different cities. This
    decoder retrieves a wider candidate set, estimates posterior mass around every candidate
    from pairwise great-circle distances, selects the highest-mass mode, and averages only
    candidates supported by that mode. It adds no trainable parameters.
    """
    m = min(top_m, scores.shape[1])
    top_scores, top_indices = scores.topk(m, dim=-1)
    vectors = gallery_unit_vectors[top_indices].float()
    cosine = torch.einsum("bmd,bnd->bmn", vectors, vectors).clamp(-1.0, 1.0)
    distances_km = 6371.0088 * torch.acos(cosine)
    base_weights = F.softmax(top_scores / vote_temperature, dim=-1)
    if kernel == "gaussian":
        support = torch.exp(-0.5 * (distances_km / radius_km).square())
    elif kernel == "hard":
        support = (distances_km <= radius_km).to(base_weights.dtype)
    else:
        raise ValueError(f"unknown geographic kernel: {kernel}")
    mode_mass = torch.einsum("bmn,bn->bm", support, base_weights)
    mode_center = mode_mass.argmax(dim=-1)
    batch_idx = torch.arange(scores.shape[0], device=scores.device)
    selected_support = support[batch_idx, mode_center]
    selected_weights = base_weights * selected_support
    selected_weights = selected_weights / selected_weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    if geometric_median_iters > 0:
        pred_vec = _weighted_geometric_median(vectors, selected_weights, geometric_median_iters)
    else:
        pred_vec = F.normalize(
            (selected_weights.unsqueeze(-1) * vectors).sum(dim=1), dim=-1, eps=1e-8
        )
    x, y, z = pred_vec.unbind(dim=-1)
    pred_lat = torch.rad2deg(torch.asin(z.clamp(-1.0, 1.0)))
    pred_lng = torch.rad2deg(torch.atan2(y, x))
    return pred_lat, pred_lng, top_indices


def _weighted_geometric_median(vectors, weights, iters=8, eps=1e-6):
    """Weiszfeld iteration for the weighted geometric median of unit vectors, done in the
    tangent chart at the current estimate then re-projected. The geometric median is the
    point estimate that minimises expected great-circle distance -- the exact minimiser of
    the challenge's median-haversine objective, and far more robust to a single far outlier
    (e.g. a Canary-Islands retrieval among mainland-Spain candidates) than the spherical mean.
    `vectors`: (B, M, 3) unit vectors; `weights`: (B, M) non-negative, sum to 1."""
    p = F.normalize((weights.unsqueeze(-1) * vectors).sum(dim=1), dim=-1, eps=1e-8)  # start = mean
    for _ in range(iters):
        cos = (vectors * p.unsqueeze(1)).sum(-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos)  # (B, M) great-circle angle to current estimate
        w = weights / (ang + eps)  # Weiszfeld reweighting
        p = F.normalize((w.unsqueeze(-1) * vectors).sum(dim=1), dim=-1, eps=1e-8)
    return p
