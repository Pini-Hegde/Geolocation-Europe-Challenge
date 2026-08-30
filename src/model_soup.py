"""
Uniform "model soup" (Wortsman et al. 2022): average the weights of N checkpoints that share
an architecture into ONE model. A single model, so it satisfies the challenge's "submit a
single final model" rule while capturing most of an ensemble's robustness.

All input checkpoints must be the same architecture (same backbone, K, cell count, embedding
dim, extra heads). Buffers (BN running stats, country-cell masks) are copied from the first
checkpoint (they are identical across seeds -- the masks are deterministic, and this project's
backbones use no BN running stats that drift meaningfully across seeds at this scale; if that
changes, re-estimate them with a forward pass over the training set).

Usage:
    python -m src.model_soup --out submission/final_model.pt \\
        outputs/experiments/T6_FULL_s1/fold-1/best_model.pt \\
        outputs/experiments/T6_FULL_s2/fold-1/best_model.pt \\
        outputs/experiments/T6_FULL_s3/fold-1/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    args = ap.parse_args()

    sds = [torch.load(c, map_location="cpu") for c in args.ckpts]
    keys = set(sds[0])
    for i, sd in enumerate(sds[1:], 1):
        if set(sd) != keys:
            raise SystemExit(f"checkpoint {i} has different keys than checkpoint 0 "
                             f"(diff: {keys.symmetric_difference(set(sd))})")
    w = args.weights or [1.0] * len(sds)
    total = sum(w)

    souped = {}
    for k in keys:
        v0 = sds[0][k]
        if not torch.is_floating_point(v0):
            souped[k] = v0  # integer buffers (masks) -- copy as-is
            continue
        acc = torch.zeros_like(v0, dtype=torch.float64)
        for wi, sd in zip(w, sds):
            acc += wi * sd[k].double()
        souped[k] = (acc / total).to(v0.dtype)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(souped, args.out)
    n = sum(v.numel() for v in souped.values() if torch.is_floating_point(v))
    print(f"souped {len(sds)} checkpoints -> {args.out}  ({n:,} float params)")


if __name__ == "__main__":
    main()
