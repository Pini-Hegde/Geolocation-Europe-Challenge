"""
Multi-partition geo-cells (T3.4, Experiment_Design.md Tier 3) -- builds K independent
per-country KD-tree partitions of the SAME points, instead of T3.1's single canonical one.

Motivation (literature.md §1, CPlaNet): T3.1's cell-accuracy bottleneck (41.8%, the ceiling
on the current 193.0km result) is a resolution/data tradeoff -- finer cells would localize
better, but there isn't enough data per cell to make them finer without overfitting (MIN=40
images/cell is already a floor). CPlaNet's fix: build several independent COARSE
partitionings (each individually as well-populated as before) and combine their predictions,
instead of using one finer partitioning. Two independent coarse decisions triangulate a
finer effective location than either one alone, without needing enough data to populate a
finer joint class space directly.

This module does NOT touch `geo_cells.py`'s canonical single partition (used by T3.1 and
every result logged before T3.4) -- it reuses `build_fold`/`build_node` with a per-partition
`np.random.Generator` (see those functions' docstrings) to get K genuinely different tree
shapes from the same points, same MIN/MAX cell-size constraints (so per-partition
data-density/difficulty stays comparable to T3.1 -- only the "multiple independent views,
then combine" mechanism is new, the isolated variable this experiment tests).

Usage:
    python -m src.protocol.geo_cells_multi
    (run from the repo root, with the project .venv active, after make_folds.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.protocol.geo_cells import N_FOLDS, MAX_CELL_SIZE, MIN_CELL_SIZE, build_fold, load_joined

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "protocol" / "geo_cells_multi"

N_PARTITIONS = 3  # K -- raised from 2 to 3, 2026-08-19, to measure the actual gain-vs-K
# curve (K=1->2 gave +10.1km/fold; is K=2->3 still worth it, or already diminishing?) before
# deciding whether K=4 is worth trying. Seeds for k=0,1 are unchanged (SEED_BASE+k is
# deterministic), so partitions 0 and 1 are byte-for-byte identical to the K=2 run -- only
# partition 2 (seed 1002) is new.
SEED_BASE = 1000  # partition k uses seed SEED_BASE + k -- offset from make_folds.py's seed=42
# and geo_cells.py's unseeded canonical build, so none of these ever coincide by accident.


def build_all_partitions(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for fold_id in range(N_FOLDS):
        for k in range(N_PARTITIONS):
            rng = np.random.default_rng(SEED_BASE + k)
            result = build_fold(df, fold_id, rng=rng)
            sizes = np.array([c["n_train"] for c in result["cells"].values()])
            print(
                f"fold {fold_id} partition {k} (seed={SEED_BASE + k}): "
                f"{result['n_cells']} cells (train-per-cell min={sizes.min()}, "
                f"median={int(np.median(sizes))}, max={sizes.max()})"
            )
            out_path = OUT_DIR / f"fold{fold_id}_p{k}_cells.json"
            out_path.write_text(
                json.dumps(
                    {
                        "meta": {
                            "fold_id": fold_id,
                            "partition": k,
                            "seed": SEED_BASE + k,
                            "min_cell_size": MIN_CELL_SIZE,
                            "max_cell_size": MAX_CELL_SIZE,
                            "method": "recursive_kdtree_median_split_per_country_randomized_axis",
                            "n_train": result["n_train"],
                            "n_val": result["n_val"],
                            "n_cells": result["n_cells"],
                        },
                        "country_trees": result["country_trees"],
                        "cells": result["cells"],
                        "assignments": result["assignments"],
                    },
                    indent=2,
                )
            )
            summary_rows.append(
                {
                    "fold": fold_id, "partition": k, "seed": SEED_BASE + k,
                    "n_cells": result["n_cells"],
                    "cell_size_min": int(sizes.min()), "cell_size_median": float(np.median(sizes)),
                    "cell_size_max": int(sizes.max()),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "partition_summary.csv", index=False)
    print(f"Artifacts written to {OUT_DIR.relative_to(REPO_ROOT)}/")


def main() -> None:
    df = load_joined()
    print(
        f"Loaded {len(df)} labelled points across {N_FOLDS} fixed folds. "
        f"Building N_PARTITIONS={N_PARTITIONS} independent partitions per fold "
        f"(same MIN={MIN_CELL_SIZE}/MAX={MAX_CELL_SIZE} as the canonical partition, "
        f"randomized split-axis choice, seeds {SEED_BASE}..{SEED_BASE + N_PARTITIONS - 1})."
    )
    build_all_partitions(df)


if __name__ == "__main__":
    main()
