"""
Country-respecting, density-adaptive geo-cell partitioning for the hierarchical
country -> within-country-cell architecture (`src/models/hierarchical_geo_classifier.py`).

Supersedes an earlier global partition (built once, ignoring country, then a soft country
bias bolted on after classification) that was tried and rolled back 2026-08-19 -- see
Progress.md's Session Log. That version's cells could straddle country borders, so ~35% of
its cell misclassifications were catastrophic cross-country errors (median ~1,250km) that a
learned soft bias only partially fixed. This version fixes it structurally instead: **every
cell belongs to exactly one country by construction**, because each country's points are
partitioned independently, before any cross-country classification decision is possible.

Method (same recursive KD-tree median split as before, just scoped per-country): for each
fold, for each of the 12 countries, split that country's training points into axis-aligned
boxes -- split whichever axis (lat/lng) has the larger spread, at the median, stop once a
node is `<= MAX_CELL_SIZE` or a further split would leave a child under `MIN_CELL_SIZE`.
Same MIN=40/MAX=120 thresholds as before; applied per-country now, so a large/dense country
(more training images) naturally gets more cells than a small/sparse one, without needing a
manually-chosen per-country cell count.

Protocol requirement (unchanged): the tree for fold i is built from ONLY fold i's training
rows -- never validation images -- so no validation-set density information leaks into cell
boundaries (same leakage class as `leakage_check.py`'s 500m clustering check). Every image's
cell assignment (train AND val) is computed from its OWN ground-truth country + coordinates
(train_labels.csv metadata, not a model prediction) -- this is a label-generation step, not
inference, so using the true country here is not a leakage path.

What it does, per fold:
1. Load that fold's rows (folds.json + train_labels.csv), split into train/val.
2. For each country: recursively partition that country's training points into cells.
   Assign each cell a running global id (0..N-1 across all 12 countries combined).
3. Compute each cell's centroid as the spherical mean of its training points.
4. Assign every image (train AND val) to a cell, by walking ITS OWN country's tree.
5. Save the per-country trees + per-cell metadata + full image->cell assignment.

Usage:
    python -m src.protocol.geo_cells
    (run from the repo root, with the project .venv active, after make_folds.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.geo.countries import COUNTRIES
from src.geo.spherical import latlon_to_unit_vector_np, unit_vector_to_latlon_np

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "geo_dataset" / "train_labels.csv"
FOLDS_JSON = REPO_ROOT / "outputs" / "protocol" / "folds" / "folds.json"
OUT_DIR = REPO_ROOT / "outputs" / "protocol" / "geo_cells"

N_FOLDS = 5
MIN_CELL_SIZE = 40
MAX_CELL_SIZE = 120


def load_joined() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_CSV)
    folds_meta = json.loads(FOLDS_JSON.read_text())
    fname_to_fold = folds_meta["filename_to_fold"]
    labels["fold"] = labels["filename"].map(fname_to_fold)
    missing = labels["fold"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} rows in train_labels.csv have no fold assignment in "
            f"{FOLDS_JSON.name} -- re-run make_folds.py first."
        )
    labels["fold"] = labels["fold"].astype(int)
    return labels


def spherical_centroid(lat_deg: np.ndarray, lng_deg: np.ndarray) -> tuple[float, float]:
    vecs = latlon_to_unit_vector_np(lat_deg, lng_deg)
    mean_vec = vecs.mean(axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)
    lat_c, lng_c = unit_vector_to_latlon_np(mean_vec[None, :])
    return float(lat_c[0]), float(lng_c[0])


class _LeafIdCounter:
    """Mutable counter shared across all 12 countries so every cell in a fold gets a unique
    global id (country 0's cells first, then country 1's, etc.)."""

    def __init__(self, start: int = 0) -> None:
        self.next_id = start

    def take(self) -> int:
        cell_id = self.next_id
        self.next_id += 1
        return cell_id


def build_node(
    lat: np.ndarray, lng: np.ndarray, idx: np.ndarray, counter: _LeafIdCounter,
    rng: "np.random.Generator | None" = None,
) -> dict:
    """Recursively partition points at positional indices `idx` (already filtered to one
    country) into a KD-tree of cells. `lat`/`lng` are that country's full training arrays.

    `rng` (T3.4, `geo_cells_multi.py`): when given, the split-axis choice is randomized
    (50/50 instead of always "larger-range axis first") so that calling this with different
    RNGs produces genuinely different, independently-shaped partitions of the SAME points --
    used to build multiple coarse partitionings for T3.4's combinatorial-cell candidate.
    `rng=None` (the default, used by every existing caller/output in this module) preserves
    the exact original deterministic behavior -- the canonical single partition T3.1 and
    every result logged before T3.4 depends on is untouched by this parameter's existence."""
    node_lat = lat[idx]
    node_lng = lng[idx]
    lat_range = float(node_lat.max() - node_lat.min())
    lng_range = float(node_lng.max() - node_lng.min())

    def make_leaf() -> dict:
        return {"leaf": True, "cell_id": counter.take(), "n_train": int(len(idx))}

    if len(idx) <= MAX_CELL_SIZE:
        return make_leaf()
    if lat_range == 0.0 and lng_range == 0.0:
        return make_leaf()

    if rng is not None:
        axis_order = list(rng.choice(["lat", "lng"], size=2, replace=False))
    else:
        axis_order = ["lat", "lng"] if lat_range >= lng_range else ["lng", "lat"]
    for axis in axis_order:
        vals = node_lat if axis == "lat" else node_lng
        if float(vals.max() - vals.min()) == 0.0:
            continue
        median = float(np.median(vals))
        left_mask = vals <= median
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        if len(left_idx) >= MIN_CELL_SIZE and len(right_idx) >= MIN_CELL_SIZE:
            return {
                "leaf": False,
                "axis": axis,
                "threshold": median,
                "left": build_node(lat, lng, left_idx, counter),
                "right": build_node(lat, lng, right_idx, counter),
            }
    # Neither axis gives a valid split (e.g. a tight cluster larger than MAX_CELL_SIZE) --
    # accept an oversized leaf rather than violate MIN_CELL_SIZE.
    return make_leaf()


def assign_cell(node: dict, lat: float, lng: float) -> int:
    """Walk one country's tree to find which leaf cell a point falls into. Only ever called
    with points known to belong to that country (see build_fold) -- there is no cross-country
    query, by construction."""
    while not node["leaf"]:
        val = lat if node["axis"] == "lat" else lng
        node = node["left"] if val <= node["threshold"] else node["right"]
    return node["cell_id"]


def collect_leaves(node: dict, out: list[dict]) -> None:
    if node["leaf"]:
        out.append(node)
    else:
        collect_leaves(node["left"], out)
        collect_leaves(node["right"], out)


def build_fold(
    df: pd.DataFrame, fold_id: int, rng: "np.random.Generator | None" = None
) -> dict:
    """`rng`: see `build_node`'s docstring -- None (default) reproduces the exact canonical
    partition every existing experiment depends on."""
    train_df = df[df["fold"] != fold_id].reset_index(drop=True)
    val_df = df[df["fold"] == fold_id].reset_index(drop=True)

    counter = _LeafIdCounter()
    country_trees: dict[str, dict] = {}
    cells: dict[int, dict] = {}

    for country in COUNTRIES:
        country_train = train_df[train_df["country"] == country].reset_index(drop=True)
        lat = country_train["lat"].to_numpy()
        lng = country_train["lng"].to_numpy()
        tree = build_node(lat, lng, np.arange(len(country_train)), counter, rng=rng)
        country_trees[country] = tree

        leaves: list[dict] = []
        collect_leaves(tree, leaves)
        train_cell_ids = np.array(
            [assign_cell(tree, la, ln) for la, ln in zip(lat, lng)], dtype=int
        )
        for leaf in leaves:
            cid = leaf["cell_id"]
            mask = train_cell_ids == cid
            lat_c, lng_c = spherical_centroid(lat[mask], lng[mask])
            cells[cid] = {
                "cell_id": cid,
                "country": country,  # exact by construction, not a majority estimate
                "n_train": int(mask.sum()),
                "n_val": 0,  # filled in below
                "centroid_lat": lat_c,
                "centroid_lng": lng_c,
            }

    # Assign every image (train AND val) to a cell by walking ITS OWN country's tree --
    # ground-truth country from train_labels.csv, not a prediction (label generation, not
    # inference -- see module docstring).
    assignments: dict[str, dict] = {}
    for split_name, split_df in (("train", train_df), ("val", val_df)):
        for row in split_df.itertuples(index=False):
            tree = country_trees[row.country]
            cid = assign_cell(tree, row.lat, row.lng)
            assignments[row.filename] = {"cell_id": cid, "split": split_name}
            if split_name == "val":
                cells[cid]["n_val"] += 1

    return {
        "fold_id": fold_id,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_cells": len(cells),
        "country_trees": country_trees,
        "cells": cells,
        "assignments": assignments,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    print(f"Loaded {len(df)} labelled points across {N_FOLDS} fixed folds, {len(COUNTRIES)} countries.")
    print(f"MIN_CELL_SIZE={MIN_CELL_SIZE}, MAX_CELL_SIZE={MAX_CELL_SIZE} (applied per-country)")

    summary_rows = []
    for fold_id in range(N_FOLDS):
        result = build_fold(df, fold_id)
        sizes = np.array([c["n_train"] for c in result["cells"].values()])
        per_country_counts = pd.Series(
            [c["country"] for c in result["cells"].values()]
        ).value_counts()
        print(
            f"fold {fold_id}: {result['n_train']} train / {result['n_val']} val -> "
            f"{result['n_cells']} cells (train-per-cell min={sizes.min()}, "
            f"median={int(np.median(sizes))}, max={sizes.max()}), "
            f"cells/country min={per_country_counts.min()} max={per_country_counts.max()}"
        )
        summary_rows.append(
            {
                "fold": fold_id,
                "n_train": result["n_train"],
                "n_val": result["n_val"],
                "n_cells": result["n_cells"],
                "cell_size_min": int(sizes.min()),
                "cell_size_median": float(np.median(sizes)),
                "cell_size_max": int(sizes.max()),
                "cells_per_country_min": int(per_country_counts.min()),
                "cells_per_country_max": int(per_country_counts.max()),
            }
        )

        out_path = OUT_DIR / f"fold{fold_id}_cells.json"
        out_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "fold_id": fold_id,
                        "min_cell_size": MIN_CELL_SIZE,
                        "max_cell_size": MAX_CELL_SIZE,
                        "method": "recursive_kdtree_median_split_per_country",
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

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "cell_summary.csv", index=False)

    lines = [
        "# Country-respecting density-adaptive geo-cells — summary",
        "",
        "Recursive KD-tree median split over (lat, lng), built independently per fold AND per "
        f"country, from that fold's training rows only (never validation). MIN_CELL_SIZE="
        f"{MIN_CELL_SIZE}, MAX_CELL_SIZE={MAX_CELL_SIZE}. Every cell belongs to exactly one "
        "country by construction -- no cell can straddle a country border.",
        "",
        "## Per-fold cell counts and sizes",
        "| fold | n_train | n_val | n_cells | min | median | max | cells/country (min-max) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['fold']} | {row['n_train']} | {row['n_val']} | {row['n_cells']} | "
            f"{row['cell_size_min']} | {row['cell_size_median']:.0f} | {row['cell_size_max']} | "
            f"{row['cells_per_country_min']}-{row['cells_per_country_max']} |"
        )
    lines += [
        "",
        "## Artifacts",
        "- `fold{i}_cells.json` — per fold: each country's split tree, every cell's id/"
        "country/centroid/train+val counts, and the full filename -> cell_id assignment.",
        "- `cell_summary.csv` — the table above, machine-readable.",
        "",
        "## Notes",
        "- Cells are built once per fold, per country, from that fold's training split only.",
        "- Centroids are spherical means (`src/geo/spherical.py`), matching the B1 baseline.",
        "- A cell's `country` field is exact (built from single-country data), not a majority "
        "estimate — this is what structurally prevents cross-country cell confusion in the "
        "hierarchical model, unlike the earlier rolled-back global-partition + soft-gate design.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines))

    print(f"Artifacts written to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
