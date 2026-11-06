"""
Geographic leakage check (Project_plan.md Phase C, step 1).

Street-level datasets are often built by driving a route and sampling frames every
few meters, which can put near-duplicate locations in both train and validation
under a naive random split -- silently inflating every validation score reported
downstream. This script checks for that before any CV strategy is locked in.

What it does:
1. Load train_labels.csv.
2. For every training point, find its nearest-neighbour distance (great-circle,
   via a haversine BallTree) to any other training point.
3. Build a "same-location cluster" graph at several candidate distance thresholds
   (points connected if within `threshold` of each other) and report cluster-size
   stats: how many points are NOT isolated (i.e. have a near-duplicate) at each
   threshold.
4. Recommend a CV strategy based on the result, and save:
   - outputs/protocol/leakage_check/nn_distance_hist.png
   - outputs/protocol/leakage_check/summary.md
   - outputs/protocol/leakage_check/results.json
   - outputs/protocol/leakage_check/clusters_<threshold>m.csv
     (filename -> cluster_id, for the threshold used to decide the strategy;
     this is the group id to feed into group-k-fold if leakage is found)

Usage:
    python -m src.protocol.leakage_check
    (run from the repo root, with the project .venv active)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "geo_dataset" / "train_labels.csv"
OUT_DIR = REPO_ROOT / "outputs" / "protocol" / "leakage_check"

EARTH_RADIUS_KM = 6371.0088

# Candidate "same location" thresholds, in meters. <500m is the threshold flagged
# in Experiment_Design.md as worth checking; the others give a sense of how the
# clustering result changes with the cutoff instead of trusting a single number.
THRESHOLDS_M = [10, 50, 100, 250, 500, 1000, 2000]

# Threshold used to decide the CV strategy and to emit the cluster-assignment file
# used for group-k-fold if leakage is found. 500m matches the value called out in
# Experiment_Design.md Part 1 Section 1.
DECISION_THRESHOLD_M = 500

# If more than this fraction of points fall into a multi-point cluster at the
# decision threshold, recommend group-stratified k-fold instead of plain
# stratified k-fold.
LEAKAGE_FRACTION_CUTOFF = 0.01


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    required = {"filename", "country", "iso", "lat", "lng"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train_labels.csv is missing expected columns: {missing}")
    return df


def build_tree(df: pd.DataFrame) -> NearestNeighbors:
    coords_rad = np.radians(df[["lat", "lng"]].to_numpy())
    tree = NearestNeighbors(algorithm="ball_tree", metric="haversine")
    tree.fit(coords_rad)
    return tree


def nearest_neighbor_distances_km(tree: NearestNeighbors, df: pd.DataFrame) -> np.ndarray:
    """For every point, distance (km) to its closest *other* point."""
    coords_rad = np.radians(df[["lat", "lng"]].to_numpy())
    # k=2: each point's nearest neighbor is itself at distance 0, so take the 2nd.
    dist_rad, _ = tree.kneighbors(coords_rad, n_neighbors=2)
    return dist_rad[:, 1] * EARTH_RADIUS_KM


def cluster_at_threshold(tree: NearestNeighbors, df: pd.DataFrame, threshold_m: float):
    """Connected components of the "within threshold_m of each other" graph."""
    coords_rad = np.radians(df[["lat", "lng"]].to_numpy())
    radius_rad = (threshold_m / 1000.0) / EARTH_RADIUS_KM
    graph = tree.radius_neighbors_graph(coords_rad, radius=radius_rad, mode="connectivity")
    n_components, labels = connected_components(graph, directed=False)
    return n_components, labels


def cluster_stats(labels: np.ndarray) -> dict:
    sizes = pd.Series(labels).value_counts()
    multi = sizes[sizes > 1]
    n_points_in_multi_cluster = int(multi.sum())
    return {
        "n_clusters": int(len(sizes)),
        "n_clusters_with_gt1_point": int(len(multi)),
        "n_points_in_multi_point_cluster": n_points_in_multi_cluster,
        "fraction_points_in_multi_point_cluster": n_points_in_multi_cluster / len(labels),
        "max_cluster_size": int(sizes.max()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labels()
    n = len(df)
    print(f"Loaded {n} labelled training points from {LABELS_CSV.relative_to(REPO_ROOT)}")

    n_exact_dupes = int(df.duplicated(subset=["lat", "lng"]).sum())
    print(f"Exact-duplicate (lat, lng) rows: {n_exact_dupes}")

    tree = build_tree(df)

    nn_km = nearest_neighbor_distances_km(tree, df)
    nn_m = nn_km * 1000.0
    print(
        "Nearest-neighbor distance (m): "
        f"min={nn_m.min():.2f}  p1={np.percentile(nn_m, 1):.2f}  "
        f"median={np.median(nn_m):.2f}  p99={np.percentile(nn_m, 99):.2f}  max={nn_m.max():.1f}"
    )

    threshold_results = {}
    for t in THRESHOLDS_M:
        _, labels = cluster_at_threshold(tree, df, t)
        stats = cluster_stats(labels)
        threshold_results[t] = stats
        print(
            f"  threshold={t:>5}m -> {stats['n_clusters_with_gt1_point']} multi-point "
            f"clusters, {stats['n_points_in_multi_point_cluster']} points affected "
            f"({100 * stats['fraction_points_in_multi_point_cluster']:.2f}%), "
            f"max cluster size={stats['max_cluster_size']}"
        )

    # Decision at the chosen threshold.
    _, decision_labels = cluster_at_threshold(tree, df, DECISION_THRESHOLD_M)
    decision_stats = threshold_results[DECISION_THRESHOLD_M]
    leakage_found = (
        decision_stats["fraction_points_in_multi_point_cluster"] > LEAKAGE_FRACTION_CUTOFF
    )
    recommendation = (
        "group-stratified k-fold (group = location cluster, stratify by country)"
        if leakage_found
        else "plain stratified k-fold (stratify by country)"
    )

    # Per-country breakdown at the decision threshold, since leakage may be
    # concentrated in a subset of countries rather than spread evenly.
    df_dec = df.copy()
    df_dec["cluster_id"] = decision_labels
    cluster_sizes = df_dec.groupby("cluster_id").size()
    df_dec["cluster_size"] = df_dec["cluster_id"].map(cluster_sizes)
    per_country = (
        df_dec.assign(in_multi_cluster=df_dec["cluster_size"] > 1)
        .groupby("country")["in_multi_cluster"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_points_in_multi_cluster", "count": "n_points"})
    )
    per_country["fraction"] = per_country["n_points_in_multi_cluster"] / per_country["n_points"]
    per_country = per_country.sort_values("fraction", ascending=False)

    # --- Save artifacts ---

    # 1. Cluster assignment CSV (needed for group-k-fold regardless of the
    #    headline decision -- cheap to save now, useful later either way).
    clusters_csv = OUT_DIR / f"clusters_{DECISION_THRESHOLD_M}m.csv"
    df_dec[["filename", "country", "cluster_id", "cluster_size"]].to_csv(
        clusters_csv, index=False
    )

    # 2. Nearest-neighbor distance histogram (log-x), with threshold markers.
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(np.log10(max(nn_m.min(), 0.1)), np.log10(nn_m.max()), 60)
    ax.hist(nn_m, bins=bins, color="#3b6ea5", edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("Nearest-neighbor distance (m, log scale)")
    ax.set_ylabel("Count of training points")
    ax.set_title("Nearest-neighbor distance between training points")
    for t in THRESHOLDS_M:
        ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(
        DECISION_THRESHOLD_M, color="red", linestyle="-", linewidth=1.2,
        label=f"decision threshold ({DECISION_THRESHOLD_M}m)",
    )
    ax.legend()
    fig.tight_layout()
    hist_path = OUT_DIR / "nn_distance_hist.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    # 3. Results JSON.
    results = {
        "n_points": n,
        "n_exact_duplicate_coords": n_exact_dupes,
        "nn_distance_m": {
            "min": float(nn_m.min()),
            "p1": float(np.percentile(nn_m, 1)),
            "median": float(np.median(nn_m)),
            "p99": float(np.percentile(nn_m, 99)),
            "max": float(nn_m.max()),
        },
        "threshold_sweep_m": threshold_results,
        "decision_threshold_m": DECISION_THRESHOLD_M,
        "leakage_fraction_cutoff": LEAKAGE_FRACTION_CUTOFF,
        "leakage_found": bool(leakage_found),
        "recommended_cv_strategy": recommendation,
        "per_country_fraction_in_multi_cluster": per_country["fraction"].round(4).to_dict(),
    }
    results_json = OUT_DIR / "results.json"
    results_json.write_text(json.dumps(results, indent=2))

    # 4. Human-readable summary.
    lines = [
        "# Geographic leakage check — summary",
        "",
        f"Training points: {n} · Exact-duplicate (lat, lng) rows: {n_exact_dupes}",
        "",
        "## Nearest-neighbor distance (meters)",
        f"min={nn_m.min():.2f}  p1={np.percentile(nn_m, 1):.2f}  "
        f"median={np.median(nn_m):.2f}  p99={np.percentile(nn_m, 99):.2f}  "
        f"max={nn_m.max():.1f}",
        "",
        "## Multi-point cluster sweep",
        "| threshold (m) | multi-point clusters | points affected | % of dataset | max cluster size |",
        "|---:|---:|---:|---:|---:|",
    ]
    for t in THRESHOLDS_M:
        s = threshold_results[t]
        lines.append(
            f"| {t} | {s['n_clusters_with_gt1_point']} | "
            f"{s['n_points_in_multi_point_cluster']} | "
            f"{100 * s['fraction_points_in_multi_point_cluster']:.2f}% | "
            f"{s['max_cluster_size']} |"
        )
    lines += [
        "",
        f"## Decision (threshold = {DECISION_THRESHOLD_M}m, cutoff = "
        f"{100 * LEAKAGE_FRACTION_CUTOFF:.1f}% of points)",
        f"- Leakage found: **{leakage_found}**",
        f"- Recommended CV strategy: **{recommendation}**",
        f"- Cluster assignments saved to `{clusters_csv.name}` "
        "(filename -> cluster_id, cluster_size).",
        "",
        "## Per-country share of points in a multi-point cluster (decision threshold)",
        "| country | points in multi-cluster | total points | fraction |",
        "|---|---:|---:|---:|",
    ]
    for country, row in per_country.iterrows():
        lines.append(
            f"| {country} | {int(row['n_points_in_multi_cluster'])} | "
            f"{int(row['n_points'])} | {row['fraction']:.4f} |"
        )
    lines += [
        "",
        f"![Nearest-neighbor distance histogram]({hist_path.name})",
        "",
    ]
    summary_md = OUT_DIR / "summary.md"
    summary_md.write_text("\n".join(lines))

    print()
    print(f"Leakage found: {leakage_found}")
    print(f"Recommended CV strategy: {recommendation}")
    print(f"Artifacts written to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
