"""Generate `predictions.csv` for `geo_dataset/holdout_public/` from a trained
`train_final.py` checkpoint.

Gallery = every labelled training image (its embeddings/cell-ids/coordinates, extracted
fresh from the checkpoint being submitted -- not cached from training). Decode = the same
hierarchy-score + retrieval-similarity scoring used throughout T4.1c/T5.1, with the decoder
family (plain top-k spherical mean, or density-mode top-k) and its hyperparameters passed on
the command line so the exact config validated by cross-validation is reproducible here.

Usage:
    python -m src.predict --exp-name T5.1_full_final --decoder plain --top-k 10 \\
        --retrieval-weight 4.0 --local-weight 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import GeoDataset, load_fold_cells_multi, load_labelled_df
from src.data.multiview import GlobalLocalTransform
from src.geo.mode_retrieval import spherical_mode_decode
from src.geo.retrieval import spherical_topk_decode
from src.models.multiview_retrieval_geo_classifier import MultiViewRetrievalGeoClassifier
from src.train_final import FULL_CELLS_DIR, FULL_FOLD_ID, ensure_full_data_cells
from src.train_multiview_retrieval_geo import extract_gallery, hierarchy_scores

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "outputs" / "experiments"
HOLDOUT_DIR = REPO_ROOT / "geo_dataset" / "holdout_public"


class HoldoutDataset(Dataset):
    """Unlabelled inference-time counterpart of `GeoDataset` -- no lat/lng/country/cell
    fields (the holdout set has none), just an image and its filename."""

    def __init__(self, filenames: list[str], transform) -> None:
        self.filenames = filenames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image = Image.open(HOLDOUT_DIR / filename).convert("RGB")
        return {"image": self.transform(image), "filename": filename}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True, help="outputs/experiments/<exp-name>, from train_final.py")
    parser.add_argument("--checkpoint-name", default="final_model.pt")
    parser.add_argument("--output", type=Path, default=None, help="default: <exp-dir>/predictions.csv")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--decoder", choices=["plain", "mode"], default="mode")
    parser.add_argument("--retrieval-weight", type=float, default=4.0)
    parser.add_argument("--local-weight", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=10, help="plain decoder only")
    parser.add_argument("--top-m", type=int, default=50, help="mode decoder only")
    parser.add_argument("--radius-km", type=float, default=150.0, help="mode decoder only")
    parser.add_argument("--kernel", choices=["gaussian", "hard"], default="hard")
    args = parser.parse_args()

    exp_dir = OUT_DIR_BASE / args.exp_name
    config = json.loads((exp_dir / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuilds outputs/protocol/geo_cells_multi_full/ from train_labels.csv if this submission's
    # copy of that cache is missing -- deterministic (fixed seeds, see geo_cells_multi.py), so
    # this reproduces the exact partition final_model.pt was trained against either way.
    ensure_full_data_cells(
        config["n_partitions"], config["min_cell_size"], config["max_cell_size"]
    )
    cell_info = load_fold_cells_multi(FULL_FOLD_ID, config["n_partitions"], cells_dir=FULL_CELLS_DIR)
    model = MultiViewRetrievalGeoClassifier(
        config["backbone"], cell_info["n_cells"], cell_info["cell_to_country"],
        embedding_dim=config["embedding_dim"], pretrained=False, dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(exp_dir / args.checkpoint_name, map_location=device))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {args.exp_name}: n_params_total={n_params:,}", flush=True)

    preprocess = config["preprocess"]
    eval_transform = GlobalLocalTransform(
        config["input_size"], preprocess["mean"], preprocess["std"],
        preprocess["crop_pct"], False, preprocess["local_fraction"],
    )

    train_df = load_labelled_df()
    gallery_ds = GeoDataset(train_df, train=False, input_size=config["input_size"], multi_cell_info=cell_info)
    gallery_ds.transform = eval_transform
    worker_args = dict(num_workers=args.num_workers, pin_memory=True)
    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False, **worker_args)
    print(f"Extracting gallery from {len(train_df)} labelled training images...", flush=True)
    gallery = extract_gallery(model, gallery_loader, device)

    holdout_filenames = sorted(p.name for p in HOLDOUT_DIR.glob("*.jpg"))
    if len(holdout_filenames) != 2400:
        print(f"WARNING: expected 2400 holdout images, found {len(holdout_filenames)}", flush=True)
    holdout_ds = HoldoutDataset(holdout_filenames, eval_transform)
    holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size, shuffle=False, **worker_args)

    rows = []
    for batch in holdout_loader:
        images = batch["image"].to(device, non_blocking=True)
        cell_logits, offset, country_logits, embedding, views = model(images)
        hierarchy = hierarchy_scores(model, country_logits, cell_logits, gallery)
        fused_similarity = embedding.float() @ gallery["embeddings"].float().T
        local_similarity = torch.einsum(
            "bvd,nd->bvn", views[:, 1:].float(), gallery["local_embeddings"].float()
        ).amax(dim=1)
        similarity = (1 - args.local_weight) * fused_similarity + args.local_weight * local_similarity
        scores = hierarchy + args.retrieval_weight * similarity
        if args.decoder == "plain":
            pred_lat, pred_lng, _ = spherical_topk_decode(scores, gallery["unit_vectors"], top_k=args.top_k)
        else:
            pred_lat, pred_lng, _ = spherical_mode_decode(
                scores, gallery["unit_vectors"], top_m=args.top_m,
                radius_km=args.radius_km, kernel=args.kernel,
            )
        pred_lat = pred_lat.cpu().numpy()
        pred_lng = pred_lng.cpu().numpy()
        for filename, lat, lng in zip(batch["filename"], pred_lat, pred_lng):
            rows.append({"filename": filename, "pred_lat": float(lat), "pred_lng": float(lng)})

    assert len(rows) == len(holdout_filenames)
    out_df = pd.DataFrame(rows, columns=["filename", "pred_lat", "pred_lng"])
    out_df = out_df.set_index("filename").loc[holdout_filenames].reset_index()  # canonical order
    output = args.output or (exp_dir / "predictions.csv")
    out_df.to_csv(output, index=False)
    print(f"Wrote {len(out_df)} predictions to {output}", flush=True)


if __name__ == "__main__":
    main()
