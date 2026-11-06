# Target parameterization verification — summary

Verifies `src/geo/spherical.py` (unit-vector target + angular loss, Experiment_Design.md
Part 1 §3) before it's trusted by any training run.

## 1. Round-trip accuracy (real data, n=11758)
lat/lng -> unit vector -> lat/lng, max error **2.84e-12 km**
(mean 4.44e-13 km) — floating-point noise, as expected.

## 2. Edge cases
- North pole: true lat=90.0, recovered lat=90.000000
- South pole: true lat=-90.0, recovered lat=-90.000000
- +-180 longitude wraparound (lng=179 vs lng=-179, true distance
  222.4km): naive raw-degree
  squared error would be **128164**
  (falsely huge), while `angular_loss` correctly stays small at
  **0.0006**. This is the concrete case
  the spherical-vector parameterization exists to fix.

## 3. Loss sanity bounds
| case | angular_loss | expected |
|---|---:|---:|
| identical vectors | 0.0000 | 0 |
| orthogonal (90°) | 1.0000 | 1 |
| antipodal (180°) | 2.0000 | 2 |
| non-unit-norm pred | 0.0000 | matches identical case |

## 4. torch vs. numpy implementation agreement (n=500 sampled)
Max abs diff — vector: 0.00e+00, lat/lng:
2.84e-14 deg, haversine: 0.00e+00 km.

## Result
**All checks passed.** `src/geo/spherical.py` is trusted for use from the Tier 1 baseline
onward.
