"""
Spherical-vector target parameterization (Experiment_Design.md Part 1, §3).

Regressing raw (lat, lng) with MSE is broken for two reasons:
  - Longitude wraps at +-180 deg: a prediction of 179 deg vs. a truth of -179 deg is 2 deg
    apart in reality but ~358^2 apart in squared-degree loss.
  - A degree of longitude is not a constant distance -- it shrinks by cos(lat) toward the
    poles, so raw-degree MSE doesn't correspond to physical (haversine) error uniformly
    across the map.

Fix (PlaNet, ISNs): regress a unit vector on the sphere,
    x = cos(lat) * cos(lng)
    y = cos(lat) * sin(lng)
    z = sin(lat)
with a cosine/angular loss (1 - cos(angle between pred and true vector)) -- a smooth,
differentiable proxy for haversine distance with no wraparound discontinuity. Convert back
to (lat, lng) via asin/atan2 only at inference/export time.

This module is used from the Tier 1 baseline onward -- every model target and every loss
computed in this project should go through these functions rather than reimplementing the
conversion inline. Two parallel implementations are provided:
  - the `*_np` functions: numpy, for precomputing dataset targets / offline analysis.
  - the `*_torch` functions + `angular_loss`: torch, differentiable, for use inside the
    training loop.

Verified against real train_labels.csv coordinates and edge cases (poles, +-180 wrap) in
`src/protocol/target_check.py` -- see outputs/protocol/target_parameterization/summary.md.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard project dependency, but keep the
    torch = None  # numpy-only half of this module importable without it (e.g. quick scripts).

EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------------------
# numpy: dataset target precomputation / offline analysis
# --------------------------------------------------------------------------------------


def latlon_to_unit_vector_np(lat_deg: np.ndarray, lng_deg: np.ndarray) -> np.ndarray:
    """(lat, lng) in degrees -> unit vector(s) of shape (..., 3): [x, y, z]."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lng = np.radians(np.asarray(lng_deg, dtype=np.float64))
    x = np.cos(lat) * np.cos(lng)
    y = np.cos(lat) * np.sin(lng)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def unit_vector_to_latlon_np(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector(s) of shape (..., 3) -> (lat_deg, lng_deg). Normalizes first, so a
    non-unit input (e.g. a raw, un-normalized model output) is handled correctly too."""
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)  # guard against a degenerate all-zero vector
    x, y, z = np.moveaxis(vec / norm, -1, 0)
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lng = np.degrees(np.arctan2(y, x))
    return lat, lng


def haversine_km_np(
    lat1: np.ndarray, lng1: np.ndarray, lat2: np.ndarray, lng2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in km between two (lat, lng) points/arrays, in degrees.
    This is the project's evaluation metric (median haversine km) -- shared here so every
    baseline / eval script uses one implementation instead of each rolling its own."""
    lat1, lng1, lat2, lng2 = (
        np.radians(np.asarray(a, dtype=np.float64)) for a in (lat1, lng1, lat2, lng2)
    )
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    a = np.clip(a, 0.0, 1.0)  # guard against tiny negative values from float rounding
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------------------
# torch: differentiable, for use inside the training loop
# --------------------------------------------------------------------------------------


def latlon_to_unit_vector_torch(lat_deg: "torch.Tensor", lng_deg: "torch.Tensor") -> "torch.Tensor":
    """(lat, lng) in degrees -> unit vector tensor of shape (..., 3). Used to build training
    targets on the fly (or precompute with the numpy version -- both must agree)."""
    lat = torch.deg2rad(lat_deg)
    lng = torch.deg2rad(lng_deg)
    x = torch.cos(lat) * torch.cos(lng)
    y = torch.cos(lat) * torch.sin(lng)
    z = torch.sin(lat)
    return torch.stack([x, y, z], dim=-1)


def unit_vector_to_latlon_torch(vec: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    """Model output vector of shape (..., 3) -> (lat_deg, lng_deg). Normalizes first, since
    a raw model output has no reason to already be unit-norm -- only used at inference/export
    time (asin/atan2 are avoided inside the training loss on purpose, see `angular_loss`)."""
    vec = torch.nn.functional.normalize(vec, dim=-1, eps=1e-12)
    x, y, z = vec.unbind(dim=-1)
    lat = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
    lng = torch.rad2deg(torch.atan2(y, x))
    return lat, lng


def angular_loss(pred_vec: "torch.Tensor", true_vec: "torch.Tensor") -> "torch.Tensor":
    """1 - cos(angle between pred and true vector), averaged over the batch.

    Smooth, differentiable proxy for haversine distance:
      - 0 when pred and true point in the same direction (perfect prediction).
      - 1 when orthogonal (90 deg apart).
      - 2 when opposite (antipodal, max possible error).
    `pred_vec` is normalized internally, so the model's raw 3-output head does not need an
    explicit normalization layer -- `true_vec` is assumed already unit-norm (as produced by
    `latlon_to_unit_vector_*`) but is normalized too as a cheap safety net.
    """
    pred_unit = torch.nn.functional.normalize(pred_vec, dim=-1, eps=1e-12)
    true_unit = torch.nn.functional.normalize(true_vec, dim=-1, eps=1e-12)
    cos_sim = (pred_unit * true_unit).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos_sim).mean()


def haversine_km_torch(
    lat1: "torch.Tensor", lng1: "torch.Tensor", lat2: "torch.Tensor", lng2: "torch.Tensor"
) -> "torch.Tensor":
    """Differentiable haversine distance in km (same formula as `haversine_km_np`) -- not
    used as the training loss (angular_loss is cheaper and avoids asin/atan2 in the backward
    pass) but handy for reporting a batch's physical error in km during validation without
    round-tripping through numpy."""
    lat1, lng1, lat2, lng2 = (torch.deg2rad(a) for a in (lat1, lng1, lat2, lng2))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng / 2) ** 2
    a = torch.clamp(a, 0.0, 1.0)
    return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a))
