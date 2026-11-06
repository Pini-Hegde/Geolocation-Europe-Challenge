"""
EDA-driven baselines B0 and B1 (Experiment_Design.md Part 1, §4 / Project_plan.md Phase C
step 4). B2 (per-country image-count-weighted random guess) is deferred -- not run here.

These are not modeling effort -- both are closed-form (a spherical mean, computed once per
fold, no training) -- they exist to split total error into "got the right country" vs.
"imprecise within the right country," before any real model is built, so later architecture
decisions (e.g. is a country-classification head worth it?) have an evidenced basis instead
of an assumed one:
  - B0 -- Global mean predictor: predict the training-fold centroid for every image.
  - B1 -- Oracle-country-centroid: given the *true* country (cheating, upper-bound only),
    predict that country's training-fold centroid. A ceiling on what perfect country
    classification alone buys us.

Both use group-stratified 5-fold CV (`outputs/protocol/folds/folds.json`) -- same protocol
every real model will be scored on, so these numbers are directly comparable to T1+ in the
results log. The "centroid" for both is a spherical mean: average the unit vectors
(`src/geo/spherical.latlon_to_unit_vector_np`) of the relevant training points and convert
back -- not a naive arithmetic mean of lat/lng degrees, which is wrong wherever it crosses
the +-180 longitude seam (same reasoning as the target parameterization itself, Experiment_
Design.md §3).

Usage:
    python -m src.protocol.eda_baselines
    (run from the repo root, with the project .venv active, after make_folds.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.geo.spherical import (
    haversine_km_np,
    latlon_to_unit_vector_np,
    unit_vector_to_latlon_np,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "geo_dataset" / "train_labels.csv"
FOLDS_JSON = REPO_ROOT / "outputs" / "protocol" / "folds" / "folds.json"
OUT_DIR = REPO_ROOT / "outputs" / "protocol" / "eda_baselines"

CITY_KM = 200  # CHALLENGE.md: "< 200 km (got the city)"
COUNTRY_KM = 750  # CHALLENGE.md: "< 750 km (got the country)"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    folds = json.loads(FOLDS_JSON.read_text())
    fold_map = folds["filename_to_fold"]
    df["fold"] = df["filename"].map(fold_map)
    missing = df["fold"].isna().sum()
    if missing:
        raise ValueError(f"{missing} rows have no fold assignment -- re-run make_folds.py.")
    df["fold"] = df["fold"].astype(int)
    return df


def spherical_centroid(lat: np.ndarray, lng: np.ndarray) -> tuple[float, float]:
    """Mean location as a spherical (unit-vector) mean, not a naive degree average --
    correct across the +-180 longitude seam, consistent with `src/geo/spherical.py`."""
    vec = latlon_to_unit_vector_np(lat, lng)
    mean_vec = vec.mean(axis=0, keepdims=True)
    lat_c, lng_c = unit_vector_to_latlon_np(mean_vec)
    return float(lat_c[0]), float(lng_c[0])


def score(err_km: np.ndarray) -> dict:
    return {
        "median_km": float(np.median(err_km)),
        "mean_km": float(np.mean(err_km)),
        "frac_lt_200km": float(np.mean(err_km < CITY_KM)),
        "frac_lt_750km": float(np.mean(err_km < COUNTRY_KM)),
    }


def run_b0(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[np.ndarray, tuple]:
    lat_c, lng_c = spherical_centroid(train_df["lat"].to_numpy(), train_df["lng"].to_numpy())
    pred_lat = np.full(len(val_df), lat_c)
    pred_lng = np.full(len(val_df), lng_c)
    err_km = haversine_km_np(val_df["lat"].to_numpy(), val_df["lng"].to_numpy(), pred_lat, pred_lng)
    return err_km, (lat_c, lng_c)


def run_b1(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    centroids = {}
    for country, g in train_df.groupby("country"):
        centroids[country] = spherical_centroid(g["lat"].to_numpy(), g["lng"].to_numpy())

    missing_countries = set(val_df["country"]) - set(centroids)
    if missing_countries:
        raise ValueError(
            f"Countries in val fold with no training centroid: {missing_countries} -- "
            "group-stratified split should prevent this."
        )

    pred_lat = val_df["country"].map(lambda c: centroids[c][0]).to_numpy()
    pred_lng = val_df["country"].map(lambda c: centroids[c][1]).to_numpy()
    err_km = haversine_km_np(val_df["lat"].to_numpy(), val_df["lng"].to_numpy(), pred_lat, pred_lng)
    return err_km, centroids


def aggregate_across_folds(per_fold: list[dict]) -> dict:
    median_kms = np.array([f["median_km"] for f in per_fold])
    return {
        "cv_median_km_mean": float(median_kms.mean()),
        "cv_median_km_std": float(median_kms.std(ddof=1)),
        "cv_mean_km_mean": float(np.mean([f["mean_km"] for f in per_fold])),
        "cv_frac_lt_200km_mean": float(np.mean([f["frac_lt_200km"] for f in per_fold])),
        "cv_frac_lt_750km_mean": float(np.mean([f["frac_lt_750km"] for f in per_fold])),
        "per_fold": per_fold,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    n_folds = int(df["fold"].max()) + 1
    print(f"Loaded {len(df)} points, {n_folds} folds from {FOLDS_JSON.relative_to(REPO_ROOT)}")

    b0_per_fold, b1_per_fold = [], []
    b1_country_err = {c: [] for c in df["country"].unique()}  # pooled per-country B1 error

    for f in range(n_folds):
        train_df = df[df["fold"] != f]
        val_df = df[df["fold"] == f]

        b0_err, _ = run_b0(train_df, val_df)
        b0_stats = score(b0_err)
        b0_per_fold.append(b0_stats)

        b1_err, _ = run_b1(train_df, val_df)
        b1_stats = score(b1_err)
        b1_per_fold.append(b1_stats)
        for country, err in zip(val_df["country"], b1_err):
            b1_country_err[country].append(err)

        print(
            f"fold {f}: B0 median={b0_stats['median_km']:.1f}km  "
            f"B1 median={b1_stats['median_km']:.1f}km"
        )

    b0_agg = aggregate_across_folds(b0_per_fold)
    b1_agg = aggregate_across_folds(b1_per_fold)

    print()
    print(
        f"B0 (global mean):        median km = {b0_agg['cv_median_km_mean']:.1f} "
        f"+/- {b0_agg['cv_median_km_std']:.1f}  |  <200km={100*b0_agg['cv_frac_lt_200km_mean']:.1f}%  "
        f"<750km={100*b0_agg['cv_frac_lt_750km_mean']:.1f}%"
    )
    print(
        f"B1 (oracle country ctr): median km = {b1_agg['cv_median_km_mean']:.1f} "
        f"+/- {b1_agg['cv_median_km_std']:.1f}  |  <200km={100*b1_agg['cv_frac_lt_200km_mean']:.1f}%  "
        f"<750km={100*b1_agg['cv_frac_lt_750km_mean']:.1f}%"
    )

    # Per-country B1 breakdown: countries with high remaining error even with the true
    # country given are the ones where within-country precision is the real problem
    # (e.g. large/elongated countries) -- informs Tier 3 grid-cell design.
    per_country_rows = []
    for country, errs in b1_country_err.items():
        errs = np.array(errs)
        per_country_rows.append(
            {
                "country": country,
                "n": len(errs),
                "median_km": float(np.median(errs)),
                "mean_km": float(np.mean(errs)),
                "frac_lt_200km": float(np.mean(errs < CITY_KM)),
            }
        )
    per_country_df = pd.DataFrame(per_country_rows).sort_values("median_km", ascending=False)
    per_country_df.to_csv(OUT_DIR / "b1_per_country.csv", index=False)

    # --- Save artifacts ---
    results = {
        "n_folds": n_folds,
        "city_km_threshold": CITY_KM,
        "country_km_threshold": COUNTRY_KM,
        "B0_global_mean": b0_agg,
        "B1_oracle_country_centroid": b1_agg,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# EDA baselines B0 / B1 — summary",
        "",
        f"Group-stratified 5-fold CV (`outputs/protocol/folds/folds.json`). Centroids are "
        "spherical means (unit-vector average, converted back), not naive degree averages.",
        "",
        "## Headline (mean ± std across folds)",
        "| baseline | median km | mean km | < 200km (city) | < 750km (country) |",
        "|---|---:|---:|---:|---:|",
        f"| B0 — global mean | {b0_agg['cv_median_km_mean']:.1f} ± {b0_agg['cv_median_km_std']:.1f} "
        f"| {b0_agg['cv_mean_km_mean']:.1f} | {100*b0_agg['cv_frac_lt_200km_mean']:.1f}% "
        f"| {100*b0_agg['cv_frac_lt_750km_mean']:.1f}% |",
        f"| B1 — oracle country centroid | {b1_agg['cv_median_km_mean']:.1f} ± "
        f"{b1_agg['cv_median_km_std']:.1f} | {b1_agg['cv_mean_km_mean']:.1f} | "
        f"{100*b1_agg['cv_frac_lt_200km_mean']:.1f}% | {100*b1_agg['cv_frac_lt_750km_mean']:.1f}% |",
        "",
        "## Reading B0 vs. B1",
        f"B1 assumes the true country is already known (oracle, upper-bound only) — the gap "
        f"between B0 and B1 is what perfect country classification alone would buy. What's "
        f"left in B1's own error is purely within-country imprecision, which no amount of "
        f"country-level classification can fix — it lower-bounds what any grid-cell / "
        f"classification-heavy approach could achieve without also getting finer localization "
        f"right.",
        "",
        "## B1 per-country breakdown (pooled across folds)",
        "Countries with the highest remaining median error, even given the true country, are "
        "where within-country precision matters most (see `b1_per_country.csv` for the full "
        "table):",
        "",
        "| country | n | median km | mean km | < 200km |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in per_country_df.head(12).iterrows():
        lines.append(
            f"| {row['country']} | {int(row['n'])} | {row['median_km']:.1f} | "
            f"{row['mean_km']:.1f} | {100*row['frac_lt_200km']:.1f}% |"
        )
    lines += [
        "",
        "*(see `b1_per_country.csv` for all countries)*",
        "",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines))

    print(f"\nArtifacts written to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
