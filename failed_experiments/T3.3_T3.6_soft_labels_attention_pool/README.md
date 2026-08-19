# T3.3 + T3.6 — haversine-smoothed soft cell labels + query attention pooling

**Status: REGRESSED, reverted 2026-08-19.** Archived here rather than deleted — this is a
real, well-reasoned negative result worth keeping for the writeup (per `Experiment_Design.md`'s
own principle: "every result is logged with reasoning, not just a number" — this applies to
failures too, see the earlier rolled-back StarNet-S1 attempt for precedent... except that one
was deleted outright; **from this experiment onward, the project convention is to archive
failed attempts here instead of deleting them.**

Full narrative, numbers, and reasoning: `Progress Tracker/Progress.md` session log
(2026-08-19 entries) and `Progress Tracker/Experiment_Design.md` Tier 3 (T3.3/T3.6 entries).
Literature basis: `Progress Tracker/literature.md` §1-2 (PIGEON, Clark et al. 2023).

## What was tried
Two changes to the T3.1 hierarchical country->cell classifier (EfficientNet-B0, 193.0±4.2km,
the working baseline this was measured against), applied together:
- **T3.3**: haversine-smoothed soft target distribution over cells (`tau=50km`) instead of
  one-hot cross-entropy, ported from PIGEON.
- **T3.6**: query attention pooling (`attention_pool.py`, this folder) — two learned queries
  (country, cell/offset) cross-attending over the backbone's spatial tokens instead of GAP,
  ported from Clark et al. 2023's "GeoDecoder", scaled down to fit the 5M-param budget.

## Result (fold-0, isolated to find which change was responsible)
| | median km | Δ vs baseline | cell acc | country acc |
|---|---|---|---|---|
| T3.1 baseline | 191.7 | — | 42.6% | 73.5% |
| T3.3-only (soft labels) | 200.5 | +8.8 | 37.7% | 72.9% |
| T3.3+T3.6 combined | 218.1 | +26.4 | 35.3% | 72.7% |
| T3.6-only (attention pool) | 230.3 | +38.6 | 37.5% | **70.9%** |

Full 5-fold (combined): **210.4 ± 7.8km** vs. T3.1's 193.0 ± 4.2km — regression confirmed
across every fold, not a single-fold fluke (cell acc landed in a tight 35.0-35.5% band vs.
T3.1's 41-43% band).

## Why it likely failed
**T3.6 (attention pooling) was the primary culprit.** It's the single worst result of the
four, worse even than the combined run, and the only condition where country accuracy also
dropped — meaning the two brand-new, randomly-initialized attention modules weren't just
failing to help cell classification, they were actively degrading a part of the model that
already worked. Most likely explanation: Clark et al. 2023's mechanism was validated with
6 shared + 2 specialized transformer layers trained on 4.72M images; this port used 1 shared
layer + 2 specialized FFNs trained on ~9.4k images/fold — probably not enough signal at that
scale for randomly-initialized attention to beat a simple pretrained-GAP-then-linear-probe.

**T3.3 (soft labels) was closer to neutral** but still not a clear win on its own (+8.8km,
one fold only, not run to full 5-fold in isolation) — not proven, not clearly ruled out
either. If revisited later, a tau sweep (this used a single untuned guess, 50km, scaled off
`OFFSET_SCALE_KM`) would be the natural next step before writing it off entirely.

## Contents
- `attention_pool.py` — the `QueryAttentionPool` module (T3.6), as it existed when removed
  from `src/models/`. No longer imported by anything in `src/` — the model reverted back to
  its original GAP-only `hierarchical_geo_classifier.py`.
- `slurm/` — every sbatch script used for this experiment (smoke test, combined 5-fold grid,
  T3.6-only and T3.3-only isolation screens). **Not directly runnable as-is** — they call
  `train_hierarchical.py` with `--pooling`/`--soft-cell-labels` flags that no longer exist in
  the reverted script; kept as an exact record of what was run, not a re-runnable pipeline.

## If revisiting this later
- T3.6 would need either much more training data (not available here — the dataset is fixed
  at ~11.7k images) or a much smaller/simpler attention mechanism with better inductive bias
  for this data regime — not a straightforward retry of the same design.
- T3.3 is the more promising thread if either is revisited: try a tau sweep on a fold-0
  screen (cheap, no architecture change) before concluding it doesn't help.
