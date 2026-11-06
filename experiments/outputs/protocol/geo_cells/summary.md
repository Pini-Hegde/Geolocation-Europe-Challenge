# Country-respecting density-adaptive geo-cells — summary

Recursive KD-tree median split over (lat, lng), built independently per fold AND per country, from that fold's training rows only (never validation). MIN_CELL_SIZE=40, MAX_CELL_SIZE=120. Every cell belongs to exactly one country by construction -- no cell can straddle a country border.

## Per-fold cell counts and sizes
| fold | n_train | n_val | n_cells | min | median | max | cells/country (min-max) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9406 | 2352 | 96 | 86 | 100 | 100 | 8-8 |
| 1 | 9407 | 2351 | 96 | 86 | 100 | 100 | 8-8 |
| 2 | 9406 | 2352 | 96 | 86 | 100 | 100 | 8-8 |
| 3 | 9407 | 2351 | 96 | 86 | 100 | 100 | 8-8 |
| 4 | 9406 | 2352 | 96 | 86 | 100 | 100 | 8-8 |

## Artifacts
- `fold{i}_cells.json` — per fold: each country's split tree, every cell's id/country/centroid/train+val counts, and the full filename -> cell_id assignment.
- `cell_summary.csv` — the table above, machine-readable.

## Notes
- Cells are built once per fold, per country, from that fold's training split only.
- Centroids are spherical means (`src/geo/spherical.py`), matching the B1 baseline.
- A cell's `country` field is exact (built from single-country data), not a majority estimate — this is what structurally prevents cross-country cell confusion in the hierarchical model, unlike the earlier rolled-back global-partition + soft-gate design.