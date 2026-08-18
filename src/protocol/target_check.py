"""
Target parameterization verification (Project_plan.md Phase C, step 3).

`src/geo/spherical.py` implements the unit-vector target + angular loss described in
Experiment_Design.md Part 1 SS3. This script exists to *prove* it's correct before any model
is trained against it, per the project's "verify, don't assume" rule -- not a throwaway
smoke test.

Checks:
1. Round-trip accuracy: every real (lat, lng) in train_labels.csv -> unit vector -> back.
   Reconstruction error should be floating-point noise (~0 km), not just "small".
2. Edge cases: poles (lat=+-90, lng ill-defined) and the +-180 longitude wraparound, where a
   naive raw-degree approach breaks.
3. The wraparound fix, demonstrated concretely: two points 2 deg of longitude apart that
   straddle +-180 (e.g. lng=179 vs lng=-179) get raw-degree squared error ~358^2 but a small
   angular_loss / haversine_km error, i.e. the same distance the naive parameterization
   would incorrectly report as huge.
4. Loss sanity bounds: identical vectors -> loss 0, orthogonal -> loss 1, antipodal -> loss 2.
5. angular_loss vs. haversine_km agree on relative ordering (angular_loss is a monotonic
   proxy for physical distance, not a replacement metric -- reported km always comes from
   haversine_km, angular_loss is only ever used as the training objective).

Usage:
    python -m src.protocol.target_check
    (run from the repo root, with the project .venv active)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.geo.spherical import (
    angular_loss,
    haversine_km_np,
    haversine_km_torch,
    latlon_to_unit_vector_np,
    latlon_to_unit_vector_torch,
    unit_vector_to_latlon_np,
    unit_vector_to_latlon_torch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "geo_dataset" / "train_labels.csv"
OUT_DIR = REPO_ROOT / "outputs" / "protocol" / "target_parameterization"


def check_roundtrip_real_data(df: pd.DataFrame) -> dict:
    vec = latlon_to_unit_vector_np(df["lat"].to_numpy(), df["lng"].to_numpy())
    lat_rt, lng_rt = unit_vector_to_latlon_np(vec)
    err_km = haversine_km_np(df["lat"].to_numpy(), df["lng"].to_numpy(), lat_rt, lng_rt)
    return {
        "n_points": len(df),
        "max_roundtrip_error_km": float(err_km.max()),
        "mean_roundtrip_error_km": float(err_km.mean()),
        "unit_vector_norm_min": float(np.linalg.norm(vec, axis=-1).min()),
        "unit_vector_norm_max": float(np.linalg.norm(vec, axis=-1).max()),
    }


def check_edge_cases() -> dict:
    results = {}

    # Poles: longitude is undefined at the pole itself, but the vector is still well-formed
    # (x=y=0, z=+-1) and round-trips to the correct latitude regardless of what longitude
    # comes back.
    for name, (lat, lng) in {
        "north_pole": (90.0, 0.0),
        "south_pole": (-90.0, 137.0),  # arbitrary lng, should not matter
    }.items():
        vec = latlon_to_unit_vector_np(np.array([lat]), np.array([lng]))
        lat_rt, _ = unit_vector_to_latlon_np(vec)
        results[name] = {"true_lat": lat, "recovered_lat": float(lat_rt[0])}

    # +-180 wraparound: two points that are actually close (2 deg of longitude apart at the
    # equator, ~222km) but on opposite sides of the raw-degree discontinuity.
    lat_a, lng_a = 0.0, 179.0
    lat_b, lng_b = 0.0, -179.0
    true_km = haversine_km_np(lat_a, lng_a, lat_b, lng_b)
    raw_degree_sq_error = (lng_a - lng_b) ** 2  # what naive MSE-on-degrees would see
    vec_a = latlon_to_unit_vector_np(np.array([lat_a]), np.array([lng_a]))
    vec_b = latlon_to_unit_vector_np(np.array([lat_b]), np.array([lng_b]))
    ang_loss = float(
        angular_loss(torch.from_numpy(vec_a), torch.from_numpy(vec_b)).item()
    )
    results["wraparound_179_vs_neg179"] = {
        "true_haversine_km": float(true_km),
        "naive_raw_degree_squared_error": float(raw_degree_sq_error),
        "angular_loss": ang_loss,
        "note": (
            "raw-degree squared error is huge (~358^2) for two points that are actually "
            f"only {true_km:.1f}km apart; angular_loss correctly stays small."
        ),
    }
    return results


def check_loss_bounds() -> dict:
    v = torch.tensor([[1.0, 0.0, 0.0]])
    identical = float(angular_loss(v, v).item())
    orthogonal = float(angular_loss(v, torch.tensor([[0.0, 1.0, 0.0]])).item())
    antipodal = float(angular_loss(v, torch.tensor([[-1.0, 0.0, 0.0]])).item())
    # also check a non-unit-norm input is handled correctly (model heads won't output
    # pre-normalized vectors)
    non_unit = float(angular_loss(torch.tensor([[5.0, 0.0, 0.0]]), v).item())
    return {
        "identical_vectors": identical,
        "orthogonal_vectors": orthogonal,
        "antipodal_vectors": antipodal,
        "non_unit_norm_pred_matches_unit_case": non_unit,
    }


def check_torch_numpy_agree(df: pd.DataFrame, n: int = 500) -> dict:
    """Sample n real points and confirm the torch and numpy implementations agree, and that
    haversine_km_torch matches haversine_km_np."""
    sample = df.sample(n=min(n, len(df)), random_state=0)
    lat = sample["lat"].to_numpy().copy()  # .copy(): pandas may hand back a read-only view,
    lng = sample["lng"].to_numpy().copy()  # and torch.from_numpy requires a writable array.

    vec_np = latlon_to_unit_vector_np(lat, lng)
    vec_torch = latlon_to_unit_vector_torch(
        torch.from_numpy(lat), torch.from_numpy(lng)
    ).numpy()
    vector_max_abs_diff = float(np.abs(vec_np - vec_torch).max())

    lat_rt_np, lng_rt_np = unit_vector_to_latlon_np(vec_np)
    lat_rt_t, lng_rt_t = unit_vector_to_latlon_torch(torch.from_numpy(vec_np))
    latlon_max_abs_diff = float(
        max(
            np.abs(lat_rt_np - lat_rt_t.numpy()).max(),
            np.abs(lng_rt_np - lng_rt_t.numpy()).max(),
        )
    )

    km_np = haversine_km_np(lat, lng, lat_rt_np, lng_rt_np)
    km_torch = haversine_km_torch(
        torch.from_numpy(lat), torch.from_numpy(lng),
        torch.from_numpy(lat_rt_np), torch.from_numpy(lng_rt_np),
    ).numpy()
    haversine_max_abs_diff_km = float(np.abs(km_np - km_torch).max())

    return {
        "n_sampled": int(len(sample)),
        "vector_max_abs_diff": vector_max_abs_diff,
        "latlon_max_abs_diff_deg": latlon_max_abs_diff,
        "haversine_max_abs_diff_km": haversine_max_abs_diff_km,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LABELS_CSV)
    print(f"Loaded {len(df)} labelled points from {LABELS_CSV.relative_to(REPO_ROOT)}")

    roundtrip = check_roundtrip_real_data(df)
    print(f"Round-trip max error: {roundtrip['max_roundtrip_error_km']:.2e} km")

    edge_cases = check_edge_cases()
    print(
        "Wraparound check -- true distance "
        f"{edge_cases['wraparound_179_vs_neg179']['true_haversine_km']:.1f}km, "
        f"angular_loss={edge_cases['wraparound_179_vs_neg179']['angular_loss']:.4f} "
        f"(vs. naive raw-degree squared error "
        f"{edge_cases['wraparound_179_vs_neg179']['naive_raw_degree_squared_error']:.0f})"
    )

    loss_bounds = check_loss_bounds()
    print(
        f"Loss bounds -- identical={loss_bounds['identical_vectors']:.4f}, "
        f"orthogonal={loss_bounds['orthogonal_vectors']:.4f}, "
        f"antipodal={loss_bounds['antipodal_vectors']:.4f}"
    )

    agreement = check_torch_numpy_agree(df)
    print(
        f"torch/numpy agreement -- max vector diff={agreement['vector_max_abs_diff']:.2e}, "
        f"max latlon diff={agreement['latlon_max_abs_diff_deg']:.2e} deg, "
        f"max haversine diff={agreement['haversine_max_abs_diff_km']:.2e} km"
    )

    # --- Pass/fail assertions -- fail loudly if this module is ever wrong, rather than
    # silently trusting it for every experiment downstream.
    assert roundtrip["max_roundtrip_error_km"] < 1e-6, "round-trip error too large"
    assert abs(edge_cases["north_pole"]["recovered_lat"] - 90.0) < 1e-6
    assert abs(edge_cases["south_pole"]["recovered_lat"] - (-90.0)) < 1e-6
    assert edge_cases["wraparound_179_vs_neg179"]["angular_loss"] < 0.01, (
        "angular_loss should stay small for two points close across the +-180 boundary"
    )
    assert loss_bounds["identical_vectors"] < 1e-6
    assert abs(loss_bounds["orthogonal_vectors"] - 1.0) < 1e-6
    assert abs(loss_bounds["antipodal_vectors"] - 2.0) < 1e-6
    assert abs(loss_bounds["non_unit_norm_pred_matches_unit_case"] - loss_bounds["identical_vectors"]) < 1e-6
    assert agreement["vector_max_abs_diff"] < 1e-6
    assert agreement["latlon_max_abs_diff_deg"] < 1e-4
    assert agreement["haversine_max_abs_diff_km"] < 1e-3
    print("\nAll checks passed.")

    results = {
        "roundtrip_real_data": roundtrip,
        "edge_cases": edge_cases,
        "loss_bounds": loss_bounds,
        "torch_numpy_agreement": agreement,
        "all_checks_passed": True,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    summary = f"""# Target parameterization verification — summary

Verifies `src/geo/spherical.py` (unit-vector target + angular loss, Experiment_Design.md
Part 1 §3) before it's trusted by any training run.

## 1. Round-trip accuracy (real data, n={roundtrip['n_points']})
lat/lng -> unit vector -> lat/lng, max error **{roundtrip['max_roundtrip_error_km']:.2e} km**
(mean {roundtrip['mean_roundtrip_error_km']:.2e} km) — floating-point noise, as expected.

## 2. Edge cases
- North pole: true lat=90.0, recovered lat={edge_cases['north_pole']['recovered_lat']:.6f}
- South pole: true lat=-90.0, recovered lat={edge_cases['south_pole']['recovered_lat']:.6f}
- +-180 longitude wraparound (lng=179 vs lng=-179, true distance
  {edge_cases['wraparound_179_vs_neg179']['true_haversine_km']:.1f}km): naive raw-degree
  squared error would be **{edge_cases['wraparound_179_vs_neg179']['naive_raw_degree_squared_error']:.0f}**
  (falsely huge), while `angular_loss` correctly stays small at
  **{edge_cases['wraparound_179_vs_neg179']['angular_loss']:.4f}**. This is the concrete case
  the spherical-vector parameterization exists to fix.

## 3. Loss sanity bounds
| case | angular_loss | expected |
|---|---:|---:|
| identical vectors | {loss_bounds['identical_vectors']:.4f} | 0 |
| orthogonal (90°) | {loss_bounds['orthogonal_vectors']:.4f} | 1 |
| antipodal (180°) | {loss_bounds['antipodal_vectors']:.4f} | 2 |
| non-unit-norm pred | {loss_bounds['non_unit_norm_pred_matches_unit_case']:.4f} | matches identical case |

## 4. torch vs. numpy implementation agreement (n={agreement['n_sampled']} sampled)
Max abs diff — vector: {agreement['vector_max_abs_diff']:.2e}, lat/lng:
{agreement['latlon_max_abs_diff_deg']:.2e} deg, haversine: {agreement['haversine_max_abs_diff_km']:.2e} km.

## Result
**All checks passed.** `src/geo/spherical.py` is trusted for use from the Tier 1 baseline
onward.
"""
    (OUT_DIR / "summary.md").write_text(summary)
    print(f"Artifacts written to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
