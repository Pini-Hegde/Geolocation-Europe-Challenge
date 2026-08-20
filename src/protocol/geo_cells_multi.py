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

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.protocol import geo_cells as geo_cells_mod
from src.protocol.geo_cells import N_FOLDS, MAX_CELL_SIZE, MIN_CELL_SIZE, build_fold, load_joined

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "protocol" / "geo_cells_multi"

N_PARTITIONS = 4  # K -- raised from 3 to 4, 2026-08-20, continuing the gain-vs-K curve
# (K=1->2 gave +10.1km/fold; K=2->3 -> T4.1c's 149.5km 5-fold best) per the K=4 question
# flagged when K=3 was raised. Seeds for k=0,1,2 are unchanged (SEED_BASE+k is
# deterministic), so partitions 0-2 are byte-for-byte identical to the K=3 run -- only
# partition 3 (seed 1003) is new.
SEED_BASE = 1000  # partition k uses seed SEED_BASE + k -- offset from make_folds.py's seed=42
# and geo_cells.py's unseeded canonical build, so none of these ever coincide by accident.


def build_all_partitions(
    df: pd.DataFrame, out_dir: Path = OUT_DIR, folds: list[int] | None = None,
    min_cell_size: int = MIN_CELL_SIZE, max_cell_size: int = MAX_CELL_SIZE,
) -> None:
    """`min_cell_size`/`max_cell_size` (screening variant, 2026-08-20): `build_node` reads
    these as bare globals off `geo_cells.py`'s own module namespace, not the names imported
    here -- so a non-default value is applied by monkeypatching `geo_cells_mod`'s attributes
    for the duration of this call, then restored, rather than by passing a parameter through
    `build_fold`. Leaves the canonical MIN=40/MAX=120 behavior byte-identical when not given."""
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = list(range(N_FOLDS)) if folds is None else folds
    orig_min, orig_max = geo_cells_mod.MIN_CELL_SIZE, geo_cells_mod.MAX_CELL_SIZE
    geo_cells_mod.MIN_CELL_SIZE, geo_cells_mod.MAX_CELL_SIZE = min_cell_size, max_cell_size
    try:
        summary_rows = []
        for fold_id in folds:
            for k in range(N_PARTITIONS):
                rng = np.random.default_rng(SEED_BASE + k)
                result = build_fold(df, fold_id, rng=rng)
                sizes = np.array([c["n_train"] for c in result["cells"].values()])
                print(
                    f"fold {fold_id} partition {k} (seed={SEED_BASE + k}): "
                    f"{result['n_cells']} cells (train-per-cell min={sizes.min()}, "
                    f"median={int(np.median(sizes))}, max={sizes.max()})"
                )
                out_path = out_dir / f"fold{fold_id}_p{k}_cells.json"
                out_path.write_text(
                    json.dumps(
                        {
                            "meta": {
                                "fold_id": fold_id,
                                "partition": k,
                                "seed": SEED_BASE + k,
                                "min_cell_size": min_cell_size,
                                "max_cell_size": max_cell_size,
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
    finally:
        geo_cells_mod.MIN_CELL_SIZE, geo_cells_mod.MAX_CELL_SIZE = orig_min, orig_max

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "partition_summary.csv", index=False)
    print(f"Artifacts written to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--min-cell-size", type=int, default=MIN_CELL_SIZE)
    parser.add_argument("--max-cell-size", type=int, default=MAX_CELL_SIZE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    df = load_joined()
    folds = args.folds if args.folds is not None else list(range(N_FOLDS))
    print(
        f"Loaded {len(df)} labelled points. Building N_PARTITIONS={N_PARTITIONS} independent "
        f"partitions for folds {folds} (MIN={args.min_cell_size}/MAX={args.max_cell_size}, "
        f"randomized split-axis choice, seeds {SEED_BASE}..{SEED_BASE + N_PARTITIONS - 1}) "
        f"-> {args.out_dir}"
    )
    build_all_partitions(
        df, out_dir=args.out_dir, folds=folds,
        min_cell_size=args.min_cell_size, max_cell_size=args.max_cell_size,
    )


if __name__ == "__main__":
    main()
