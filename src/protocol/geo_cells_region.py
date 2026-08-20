"""Coarse per-country "region" partition -- T4.1d, the T3.5/ISNs-style intermediate
hierarchy level (`literature.md` §1/§5's top remaining untried candidate).

Motivation: cell-classification accuracy has been stuck at ~42-44% across four different
architecture generations (T3.1 193.0km / T3.4 182.9km / T4.1 166.85km / T4.1c 149.5km) even
though the surrounding mechanism (hard masking -> combinatorial blending -> soft retrieval ->
contrastive refinement) kept improving the final score. This module builds ONE additional,
much coarser per-country partition (MIN/MAX far larger than `geo_cells.py`'s 40/120, so each
country gets only ~2-3 regions instead of ~8 fine cells) -- an easier classification target
that should be resolvable at much higher accuracy, giving the retrieval score a better-
conditioned coarse signal before the fine cell picks. Intentionally targets the exact large/
sparse countries (Norway, Spain, Turkey) flagged as worst in the B1 baseline
(`Experiment_Design.md` §4) -- a flat 8-cells/country split is coarse for them regardless of
overall country size, since they're geographically large without proportionally more data.

Deliberately a SELF-CONTAINED recursive splitter (own build_node/build_fold, own MIN/MAX
constants), NOT a call into `geo_cells.py` with different constants -- that module's
MIN_CELL_SIZE/MAX_CELL_SIZE are plain module globals read directly inside `build_node`, not
parameters, so overriding them would mean monkeypatching a working baseline file's internals
mid-run. Duplicating ~50 lines here keeps `geo_cells.py` (T3.1's dependency, still used by
every fine partition) completely untouched, consistent with this project's established
convention (see `geo_cells_multi.py`'s own header note, `failed_experiments/README`s).

Usage:
    python -m src.protocol.geo_cells_region
    (run from the repo root, with the project .venv active, after make_folds.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.geo.countries import COUNTRIES
from src.protocol.geo_cells import N_FOLDS, load_joined, spherical_centroid

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "protocol" / "geo_cells_region"

# Far coarser than geo_cells.py's 40/120 -- targets ~2-3 regions per country (roughly
# 700-950 fold-training images/country in this dataset) instead of ~8 fine cells/country.
# Not tuned via search (time-boxed decision, consistent with T3.3's tau being an untuned
# guess) -- the goal is "genuinely coarser, easier to classify," not a specific region count.
MIN_REGION_SIZE = 150
MAX_REGION_SIZE = 400


class _LeafIdCounter:
    def __init__(self, start: int = 0) -> None:
        self.next_id = start

    def take(self) -> int:
        cell_id = self.next_id
        self.next_id += 1
        return cell_id


def build_region_node(lat: np.ndarray, lng: np.ndarray, idx: np.ndarray, counter: _LeafIdCounter) -> dict:
    """Same recursive median-split logic as `geo_cells.py::build_node`, but with this
    module's own (much coarser) MIN/MAX -- see module docstring for why this isn't a call
    into that function with overridden globals."""
    node_lat, node_lng = lat[idx], lng[idx]
    lat_range = float(node_lat.max() - node_lat.min())
    lng_range = float(node_lng.max() - node_lng.min())

    def make_leaf() -> dict:
        return {"leaf": True, "cell_id": counter.take(), "n_train": int(len(idx))}

    if len(idx) <= MAX_REGION_SIZE:
        return make_leaf()
    if lat_range == 0.0 and lng_range == 0.0:
        return make_leaf()

    axis_order = ["lat", "lng"] if lat_range >= lng_range else ["lng", "lat"]
    for axis in axis_order:
        vals = node_lat if axis == "lat" else node_lng
        if float(vals.max() - vals.min()) == 0.0:
            continue
        median = float(np.median(vals))
        left_mask = vals <= median
        left_idx, right_idx = idx[left_mask], idx[~left_mask]
        if len(left_idx) >= MIN_REGION_SIZE and len(right_idx) >= MIN_REGION_SIZE:
            return {
                "leaf": False, "axis": axis, "threshold": median,
                "left": build_region_node(lat, lng, left_idx, counter),
                "right": build_region_node(lat, lng, right_idx, counter),
            }
    return make_leaf()


def assign_region(node: dict, lat: float, lng: float) -> int:
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


def build_fold(df: pd.DataFrame, fold_id: int) -> dict:
    train_df = df[df["fold"] != fold_id].reset_index(drop=True)
    val_df = df[df["fold"] == fold_id].reset_index(drop=True)

    counter = _LeafIdCounter()
    country_trees: dict[str, dict] = {}
    cells: dict[int, dict] = {}

    for country in COUNTRIES:
        country_train = train_df[train_df["country"] == country].reset_index(drop=True)
        lat = country_train["lat"].to_numpy()
        lng = country_train["lng"].to_numpy()
        tree = build_region_node(lat, lng, np.arange(len(country_train)), counter)
        country_trees[country] = tree

        leaves: list[dict] = []
        collect_leaves(tree, leaves)
        train_region_ids = np.array(
            [assign_region(tree, la, ln) for la, ln in zip(lat, lng)], dtype=int
        )
        for leaf in leaves:
            cid = leaf["cell_id"]
            mask = train_region_ids == cid
            lat_c, lng_c = spherical_centroid(lat[mask], lng[mask])
            cells[cid] = {
                "cell_id": cid, "country": country, "n_train": int(mask.sum()),
                "n_val": 0, "centroid_lat": lat_c, "centroid_lng": lng_c,
            }

    assignments: dict[str, dict] = {}
    for split_name, split_df in (("train", train_df), ("val", val_df)):
        for row in split_df.itertuples(index=False):
            tree = country_trees[row.country]
            cid = assign_region(tree, row.lat, row.lng)
            assignments[row.filename] = {"cell_id": cid, "split": split_name}
            if split_name == "val":
                cells[cid]["n_val"] += 1

    return {
        "fold_id": fold_id, "n_train": int(len(train_df)), "n_val": int(len(val_df)),
        "n_cells": len(cells), "country_trees": country_trees,
        "cells": cells, "assignments": assignments,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    print(f"Loaded {len(df)} labelled points across {N_FOLDS} fixed folds, {len(COUNTRIES)} countries.")
    print(f"MIN_REGION_SIZE={MIN_REGION_SIZE}, MAX_REGION_SIZE={MAX_REGION_SIZE} (applied per-country)")

    summary_rows = []
    for fold_id in range(N_FOLDS):
        result = build_fold(df, fold_id)
        sizes = np.array([c["n_train"] for c in result["cells"].values()])
        per_country = {c: 0 for c in COUNTRIES}
        for cell in result["cells"].values():
            per_country[cell["country"]] += 1
        print(
            f"fold {fold_id}: {result['n_cells']} regions total "
            f"(train-per-region min={sizes.min()}, median={int(np.median(sizes))}, max={sizes.max()}); "
            f"regions/country: {per_country}"
        )
        out_path = OUT_DIR / f"fold{fold_id}_cells.json"
        out_path.write_text(json.dumps({
            "meta": {
                "fold_id": fold_id, "min_region_size": MIN_REGION_SIZE,
                "max_region_size": MAX_REGION_SIZE,
                "method": "recursive_kdtree_median_split_per_country_coarse",
                "n_train": result["n_train"], "n_val": result["n_val"], "n_cells": result["n_cells"],
            },
            "country_trees": result["country_trees"], "cells": result["cells"],
            "assignments": result["assignments"],
        }, indent=2))
        summary_rows.append({
            "fold": fold_id, "n_regions": result["n_cells"],
            "region_size_min": int(sizes.min()), "region_size_median": float(np.median(sizes)),
            "region_size_max": int(sizes.max()),
        })

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "region_summary.csv", index=False)
    print(f"Artifacts written to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
