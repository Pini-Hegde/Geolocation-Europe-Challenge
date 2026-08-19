"""
Aggregate a real cv_summary.json from per-fold results that were trained as SEPARATE parallel
SLURM jobs (one fold per job, same `--exp-name`) rather than one job looping over all folds.

`src/train_hierarchical.py`'s own cv_summary.json aggregation only fires when a single run
actually covers more than one fold -- a single-fold job intentionally does NOT write it (see
that script's comment), to avoid one job's single-fold result silently overwriting another's
under a misleading "cv_summary.json" filename. This script is the correct way to produce the
real aggregate afterwards, once all the fold jobs for one `exp_name` have finished: it reads
each `fold{i}/fold_result.json` directly off disk and computes the same mean/std the
in-process aggregation would have, over however many folds are actually present.

Usage:
    python -m src.aggregate_fold_results --exp-name T3.1_hier_efficientnet_b0_224_5fold
    python -m src.aggregate_fold_results --exp-name T3.1_hier_efficientnet_b0_224_5fold --folds 0 1 2 3 4
        (--folds restricts to specific fold ids; default is "whatever fold{i}/ dirs exist")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "outputs" / "experiments"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="restrict to these fold ids; default is every fold{i}/fold_result.json present",
    )
    args = parser.parse_args()

    exp_dir = OUT_DIR_BASE / args.exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"{exp_dir} does not exist")

    if args.folds is not None:
        fold_ids = args.folds
    else:
        fold_ids = sorted(
            int(p.name.removeprefix("fold"))
            for p in exp_dir.glob("fold*")
            if p.is_dir() and (p / "fold_result.json").exists()
        )
    if not fold_ids:
        raise FileNotFoundError(f"no fold{{i}}/fold_result.json found under {exp_dir}")

    fold_results = {}
    for fold_id in fold_ids:
        result_path = exp_dir / f"fold{fold_id}" / "fold_result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"missing {result_path} -- that fold's job may still be running")
        fold_results[fold_id] = json.loads(result_path.read_text())

    medians = np.array([r["best_val_median_km"] for r in fold_results.values()])
    means = np.array([r["mean_km"] for r in fold_results.values()])
    cell_accs = np.array([r["cell_top1_acc"] for r in fold_results.values()])
    country_accs = np.array([r["country_top1_acc"] for r in fold_results.values()])

    config_path = exp_dir / "config.json"
    backbone = json.loads(config_path.read_text())["backbone"] if config_path.exists() else None
    input_size = json.loads(config_path.read_text())["input_size"] if config_path.exists() else None

    summary = {
        "backbone": backbone,
        "input_size": input_size,
        "folds_run": fold_ids,
        "cv_median_km_mean": float(medians.mean()),
        "cv_median_km_std": float(medians.std(ddof=1)) if len(medians) > 1 else 0.0,
        "cv_mean_km_mean": float(means.mean()),
        "cv_cell_top1_acc_mean": float(cell_accs.mean()),
        "cv_country_top1_acc_mean": float(country_accs.mean()),
        "per_fold": {str(k): v for k, v in fold_results.items()},
        "aggregated_from_separate_jobs": True,
    }
    (exp_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"{args.exp_name}: {len(fold_ids)} fold(s) {fold_ids} -> median km = "
        f"{summary['cv_median_km_mean']:.1f} +/- {summary['cv_median_km_std']:.1f}, "
        f"mean km = {summary['cv_mean_km_mean']:.1f}, "
        f"cell acc = {100*summary['cv_cell_top1_acc_mean']:.1f}%, "
        f"country acc = {100*summary['cv_country_top1_acc_mean']:.1f}%"
    )
    print(f"Written to {(exp_dir / 'cv_summary.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
