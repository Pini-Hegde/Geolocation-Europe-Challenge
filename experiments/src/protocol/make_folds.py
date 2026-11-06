"""
Fixed CV fold generation (Project_plan.md Phase C, step 2).

Consumes the leakage check's cluster assignments (`clusters_500m.csv`) and generates
a single, fixed group-stratified 5-fold split of `train_labels.csv`:
  - group = 500m location cluster (a whole cluster always lands in one fold, so a
    near-duplicate location can never straddle train/val within a fold);
  - stratified by `country`, so each fold keeps roughly the overall country mix.

This is generated once and reused by every experiment from here on (Experiment_Design.md
§2) so that results across the results log are comparable on identical splits, rather than
each run drawing its own random split.

What it does:
1. Load train_labels.csv and clusters_500m.csv, join on filename.
2. Run sklearn's StratifiedGroupKFold(n_splits=5) with groups=cluster_id, y=country.
3. Verify group integrity (no cluster split across folds) and report per-fold,
   per-country counts so we can eyeball how balanced the stratification actually came out
   (group-stratification is a best-effort constraint solve, not exact, especially for the
   few large Iceland/Norway clusters).
4. Save:
   - outputs/protocol/folds/folds.json (filename -> fold id, 0-4, + generation metadata)
   - outputs/protocol/folds/fold_country_counts.csv (per-fold x per-country counts, for the
     write-up / a sanity glance)
   - outputs/protocol/folds/summary.md

Usage:
    python -m src.protocol.make_folds
    (run from the repo root, with the project .venv active, after leakage_check.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = REPO_ROOT / "geo_dataset" / "train_labels.csv"
CLUSTERS_CSV = REPO_ROOT / "outputs" / "protocol" / "leakage_check" / "clusters_500m.csv"
OUT_DIR = REPO_ROOT / "outputs" / "protocol" / "folds"

N_FOLDS = 5
SEED = 42  # fixed so folds.json is reproducible from scratch if ever regenerated
GROUP_THRESHOLD_M = 500


def load_joined() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_CSV)
    clusters = pd.read_csv(CLUSTERS_CSV)
    df = labels.merge(
        clusters[["filename", "cluster_id", "cluster_size"]], on="filename", how="left"
    )
    missing = df["cluster_id"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} rows in train_labels.csv have no matching cluster_id in "
            f"{CLUSTERS_CSV.name} -- re-run leakage_check.py first."
        )
    df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def assign_folds(df: pd.DataFrame) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(df), -1, dtype=int)
    for fold_id, (_, val_idx) in enumerate(
        sgkf.split(df, y=df["country"], groups=df["cluster_id"])
    ):
        fold[val_idx] = fold_id
    assert (fold >= 0).all(), "every row must be assigned to a fold"
    return fold


def check_group_integrity(df: pd.DataFrame) -> None:
    """Assert no cluster (group) is split across more than one fold."""
    folds_per_cluster = df.groupby("cluster_id")["fold"].nunique()
    bad = folds_per_cluster[folds_per_cluster > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} cluster(s) are split across multiple folds -- group-k-fold "
            f"invariant violated: {bad.index.tolist()[:10]}"
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined()
    n = len(df)
    n_groups = df["cluster_id"].nunique()
    print(f"Loaded {n} labelled points, {n_groups} location clusters (500m).")

    df["fold"] = assign_folds(df)
    check_group_integrity(df)
    print("Group integrity check passed: no cluster split across folds.")

    fold_sizes = df["fold"].value_counts().sort_index()
    print("Fold sizes:", fold_sizes.to_dict())

    # Per-fold x per-country counts, for a stratification sanity check.
    counts = df.pivot_table(
        index="country", columns="fold", values="filename", aggfunc="count", fill_value=0
    ).astype(int)
    counts.columns = [f"fold_{c}" for c in counts.columns]
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False)

    # Max deviation of any (country, fold) share from that country's overall share,
    # as a quick single-number stratification-quality check.
    country_totals = df.groupby("country").size()
    overall_share = 1.0 / N_FOLDS
    max_dev = 0.0
    for country, row in counts.iterrows():
        for f in range(N_FOLDS):
            share = row[f"fold_{f}"] / country_totals[country]
            max_dev = max(max_dev, abs(share - overall_share))
    print(f"Max per-(country, fold) share deviation from ideal {overall_share:.3f}: {max_dev:.3f}")

    # --- Save artifacts ---

    # 1. folds.json -- the canonical artifact every downstream experiment reads.
    folds_json = {
        "meta": {
            "n_folds": N_FOLDS,
            "seed": SEED,
            "method": "sklearn.model_selection.StratifiedGroupKFold",
            "group_col": "cluster_id",
            "group_threshold_m": GROUP_THRESHOLD_M,
            "stratify_col": "country",
            "source_labels_csv": str(LABELS_CSV.relative_to(REPO_ROOT)),
            "source_clusters_csv": str(CLUSTERS_CSV.relative_to(REPO_ROOT)),
            "n_images": n,
            "n_groups": int(n_groups),
            "fold_sizes": {int(k): int(v) for k, v in fold_sizes.items()},
            "max_country_fold_share_deviation": round(max_dev, 4),
        },
        # filename -> fold id (0..N_FOLDS-1) it belongs to when *held out* as validation.
        # For fold i: val = {f: fold[f] == i}, train = {f: fold[f] != i}.
        "filename_to_fold": dict(zip(df["filename"], df["fold"].astype(int))),
    }
    folds_path = OUT_DIR / "folds.json"
    folds_path.write_text(json.dumps(folds_json, indent=2))

    # 2. Per-fold x per-country counts CSV.
    counts_path = OUT_DIR / "fold_country_counts.csv"
    counts.to_csv(counts_path)

    # 3. Human-readable summary.
    lines = [
        "# Fixed CV folds — summary",
        "",
        f"{n} labelled points, {n_groups} location clusters (500m threshold), "
        f"split into {N_FOLDS} group-stratified folds (seed={SEED}).",
        "",
        "## Fold sizes",
        "| fold | n images |",
        "|---:|---:|",
    ]
    for f, cnt in fold_sizes.items():
        lines.append(f"| {f} | {cnt} |")
    lines += [
        "",
        "## Group integrity",
        "Every 500m location cluster lands entirely within a single fold (checked, passed).",
        "",
        f"## Stratification quality",
        f"Max deviation of any (country, fold) share from the ideal {overall_share:.3f} "
        f"(1/{N_FOLDS}): **{max_dev:.3f}**. Some slack is expected — group-stratification is a "
        "best-effort solve, not exact, especially for countries with a few large clusters "
        "(e.g. Iceland/Norway, see leakage_check summary).",
        "",
        "## Per-country x per-fold counts",
        "See `fold_country_counts.csv` for the full table.",
        "",
        "## Artifacts",
        "- `folds.json` — canonical filename -> fold mapping + generation metadata, "
        "consumed by every training/eval script from here on.",
        "- `fold_country_counts.csv` — per-country x per-fold image counts.",
        "",
    ]
    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines))

    print(f"Artifacts written to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
