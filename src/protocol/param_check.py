"""
Automated parameter-compliance checker (Experiment_Design.md Part 1, §5).

The challenge's hard constraint is <= 5,000,000 parameters in the *submitted* model.
Reported param counts from papers/model zoos aren't trustworthy on their own -- they often
assume a different input resolution or a classifier head we're not using -- so this
instantiates the actual model class this project trains, at the actual input resolution,
and counts real `sum(p.numel())`. Every training script imports `check_compliance` and
calls it before training starts, not after -- this is a gate, not a post-hoc report.

Usage (standalone, to check a candidate before wiring it into training):
    python -m src.protocol.param_check --backbone mobilevit_xs.cvnets_in1k
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.dataset import INPUT_SIZE
from src.models.backbone_regressor import GeoBackboneRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "protocol" / "param_check"
PARAM_LIMIT = 5_000_000


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def check_compliance(
    backbone_name: str, pretrained: bool = True, input_size: int = INPUT_SIZE
) -> dict:
    """Instantiate the real model at the real input resolution and verify param count.
    Raises AssertionError if over budget -- callers should let this propagate (fail loudly,
    don't train a non-compliant model)."""
    model = GeoBackboneRegressor(backbone_name, pretrained=pretrained)
    n_params = count_params(model)
    n_backbone = count_params(model.backbone)
    n_head = count_params(model.head)

    # Confirm the model actually runs at the claimed input resolution (catches "reported
    # for a different input size" surprises the docstring warns about) before trusting the
    # param count as "the real one".
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, input_size, input_size)
        out = model(dummy)
    assert out.shape == (1, 3), f"expected output shape (1, 3), got {tuple(out.shape)}"

    compliant = n_params <= PARAM_LIMIT
    result = {
        "backbone_name": backbone_name,
        "pretrained": pretrained,
        "input_size": input_size,
        "n_params_total": n_params,
        "n_params_backbone": n_backbone,
        "n_params_head": n_head,
        "param_limit": PARAM_LIMIT,
        "compliant": compliant,
    }
    assert compliant, (
        f"{backbone_name} has {n_params:,} params at input_size={input_size}, "
        f"over the {PARAM_LIMIT:,} limit."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="mobilevit_xs.cvnets_in1k")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = check_compliance(
        args.backbone, pretrained=not args.no_pretrained, input_size=args.input_size
    )

    print(
        f"{result['backbone_name']}: {result['n_params_total']:,} params "
        f"(backbone {result['n_params_backbone']:,} + head {result['n_params_head']:,}) "
        f"at {result['input_size']}px -- "
        f"{'COMPLIANT' if result['compliant'] else 'OVER BUDGET'} "
        f"(limit {result['param_limit']:,})"
    )

    safe_name = args.backbone.replace("/", "_")
    out_path = OUT_DIR / f"{safe_name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Result written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
