"""
GeoCLIP-style continuous location encoder (Cepeda et al., 2023), shrunk to fit this
project's 5M-parameter budget.

Maps a (lat, lng) coordinate to an L2-normalized embedding in the same space as the
image embedding, so an image can be scored against *any* coordinate -- not only the
labelled training points -- via a dot product. This gives the retrieval decoder a smooth
spatial prior that generalizes into the gaps between training images (the large-sparse-
country failure mode: France / Spain / Norway).

Pipeline per GeoCLIP Sec 3.1.1:
  (lat, lng) --Equal Earth Projection--> 2D  --random Fourier features at M scales-->
  per-scale MLP --sum--> L2-normalized embedding.

The random projection matrices R_i are fixed buffers (0 trainable params). Only the M
small MLPs are trainable (~2e5 params at the default sizes), leaving room for the
EfficientNet-B0 backbone + K=3 cell heads under the cap.

No external data: the encoder is trained only on the provided labels' coordinates
(aligned to their images) plus synthetic coordinates sampled from those same labels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Equal Earth projection polynomial constants (Savric et al. 2018; GeoCLIP Eq. 1).
_A1, _A2, _A3, _A4 = 1.340264, -0.081106, 0.000893, 0.003796


def equal_earth_projection(lat_deg: torch.Tensor, lng_deg: torch.Tensor) -> torch.Tensor:
    """(lat, lng) degrees -> (x, y) in the Equal Earth projection, roughly scaled to
    [-1, 1] on each axis. Counteracts the area distortion of raw lat/lng near the poles."""
    lat = torch.deg2rad(lat_deg.to(torch.float32))
    lng = torch.deg2rad(lng_deg.to(torch.float32))
    theta = torch.asin((math.sqrt(3.0) / 2.0) * torch.sin(lat))
    t2 = theta * theta
    t6 = t2 * t2 * t2
    x = (2.0 * math.sqrt(3.0) * lng * torch.cos(theta)) / (
        3.0 * (9.0 * _A4 * t6 * t2 + 7.0 * _A3 * t6 + 3.0 * _A2 * t2 + _A1)
    )
    y = _A4 * theta * t6 * t2 + _A3 * theta * t6 + _A2 * theta * t2 + _A1 * theta
    # Normalize to approx [-1, 1] using the projection's extent at the map edges.
    return torch.stack([x / 2.6667, y / 1.3333], dim=-1)


class LocationEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        n_scales: int = 3,
        fourier_features: int = 64,
        hidden_dim: int = 128,
        sigma_min: float = 1.0,
        sigma_max: float = 256.0,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales
        sigmas = [
            2.0 ** (
                math.log2(sigma_min)
                + (i / max(1, n_scales - 1)) * (math.log2(sigma_max) - math.log2(sigma_min))
            )
            for i in range(n_scales)
        ]
        self.mlps = nn.ModuleList()
        for i, sigma in enumerate(sigmas):
            r = torch.randn(fourier_features, 2) * sigma
            self.register_buffer(f"rff_{i}", r, persistent=True)
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(2 * fourier_features, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, embedding_dim),
                )
            )

    def forward(self, lat_deg: torch.Tensor, lng_deg: torch.Tensor) -> torch.Tensor:
        xy = equal_earth_projection(lat_deg, lng_deg)  # (..., 2)
        out = 0.0
        for i, mlp in enumerate(self.mlps):
            r = getattr(self, f"rff_{i}")  # (F, 2)
            proj = 2.0 * math.pi * (xy @ r.T)  # (..., F)
            gamma = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (..., 2F)
            out = out + mlp(gamma)
        return F.normalize(out, dim=-1, eps=1e-8)


def sample_synthetic_coords(
    lat_deg: torch.Tensor, lng_deg: torch.Tensor, n: int, jitter_km: float = 20.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw `n` synthetic coordinates by resampling the given (labelled) coordinates with
    Gaussian jitter -- stays on the data manifold (no ocean / wrong-country points) while
    filling gaps between training images. Used both for extra InfoNCE negatives at train
    time and (optionally) as extra decode candidates at inference."""
    idx = torch.randint(0, len(lat_deg), (n,), generator=generator)
    base_lat = lat_deg[idx].to(torch.float32)
    base_lng = lng_deg[idx].to(torch.float32)
    km_per_deg = 111.195
    dlat = torch.randn(n, generator=generator) * (jitter_km / km_per_deg)
    dlng = torch.randn(n, generator=generator) * (
        jitter_km / (km_per_deg * torch.cos(torch.deg2rad(base_lat)).clamp_min(0.1))
    )
    return base_lat + dlat, base_lng + dlng
