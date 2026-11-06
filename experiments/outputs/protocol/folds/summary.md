# Fixed CV folds — summary

11758 labelled points, 10068 location clusters (500m threshold), split into 5 group-stratified folds (seed=42).

## Fold sizes
| fold | n images |
|---:|---:|
| 0 | 2352 |
| 1 | 2351 |
| 2 | 2352 |
| 3 | 2351 |
| 4 | 2352 |

## Group integrity
Every 500m location cluster lands entirely within a single fold (checked, passed).

## Stratification quality
Max deviation of any (country, fold) share from the ideal 0.200 (1/5): **0.001**. Some slack is expected — group-stratification is a best-effort solve, not exact, especially for countries with a few large clusters (e.g. Iceland/Norway, see leakage_check summary).

## Per-country x per-fold counts
See `fold_country_counts.csv` for the full table.

## Artifacts
- `folds.json` — canonical filename -> fold mapping + generation metadata, consumed by every training/eval script from here on.
- `fold_country_counts.csv` — per-country x per-fold image counts.
