# Europe Geolocation Challenge — submission

Predicts `(lat, lng)` for a 512x512 European street-level photo. Architecture: EfficientNet-B0
backbone (ImageNet-pretrained, `timm`) feeding a country classifier, three independent
per-country geo-cell classifiers ("K=3 coarse partitions" -- CPlaNet-style), a local-offset
regressor, and a 256-d geographic embedding trained with a supervised contrastive + smooth
geo-distribution loss. Inference retrieves the nearest labelled training images in embedding
+ hierarchy-score space and decodes the prediction as a density-weighted spherical mean over
the strongest coherent geographic mode among the top candidates (see the writeup for the
full method and the ablations that led here).

**Final params: 4,999,042 / 5,000,000 (compliant).**

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Expects the repo layout as provided:
```
geo_dataset/train/                # labelled training images
geo_dataset/train_labels.csv
geo_dataset/holdout_public/       # 2,400 unlabelled evaluation images
```

## Reproducing the result

1. **Fixed folds + geo-cells** (deterministic, only needs to run once):
   ```bash
   python -m src.protocol.make_folds
   python -m src.protocol.geo_cells_multi
   ```
2. **Train the submitted model** (all labelled data, no held-out split -- see the writeup for
   why: hyperparameters/epoch count were chosen from a separate 5-fold CV screen, not from
   this run):
   ```bash
   python -m src.train_final \
     --backbone efficientnet_b0.ra4_e3600_r224_in1k --n-partitions 3 --input-size 288 \
     --epochs 15 --batch-size 32 --backbone-lr 2e-5 --head-lr 2e-4 \
     --contrastive-weight 0.2 \
     --init-checkpoint outputs/experiments/T4.1c_geo_supcon_5fold/fold0/best_model.pt \
     --exp-name T5.3_full_final
   ```
   Writes `outputs/experiments/T5.3_full_final/final_model.pt`.
   **Training time:** ~31.5 minutes (15 epochs) on one RTX 4000 Ada (20GB), warm-started from
   a 5-fold-CV checkpoint (see the writeup for why: hyperparameters/epoch count were chosen on
   that separate CV screen, not on this run, which has no held-out labels to validate against).
3. **Generate `predictions.csv`** for `geo_dataset/holdout_public/`:
   ```bash
   python -m src.predict --exp-name T5.3_full_final --decoder mode \
     --retrieval-weight 4.0 --local-weight 0.25 --top-m 50 --radius-km 150 --kernel hard
   ```
   Writes `outputs/experiments/T5.3_full_final/predictions.csv` (a copy is also at
   `predictions.csv` in this submission's root), header `filename,pred_lat,pred_lng`, one row
   per holdout image. This exact command was smoke-tested end-to-end from a fresh copy of this
   submission directory before packaging.

## Validated performance (cross-validation, not the holdout set)

5-fold CV, honest fixed decoder (no per-fold decoder selection): **median 150.6 km**, mean
447.5 km, 55% of images within 200km ("got the city"), 82% within 750km ("got the country").
See the writeup for the full per-country breakdown and what was tried.

## Files in this submission

- `train_final.py`, `train_multiview_retrieval_geo.py`, `train_retrieval_geo.py` — training
  scripts (the last two are the fold-CV screening scripts the final config was chosen from).
- `predict.py` — generates `predictions.csv` for the holdout set from a trained checkpoint.
- `models/`, `data/`, `geo/`, `protocol/` — architecture, dataset, spherical-geometry and
  fold/geo-cell-building code shared by every script above.
- `final_model.pt`, `config.json`, `predictions.csv` — root-level convenience copies of the
  files `train_final.py`/`predict.py` write to `outputs/experiments/T5.3_full_final/`.
- `outputs/experiments/T5.3_full_final/` — the same files, in the layout the scripts expect
  (`--exp-name T5.3_full_final` reads/writes here).
- `outputs/experiments/T4.1c_geo_supcon_5fold/fold0/best_model.pt` — the 5-fold-CV checkpoint
  `final_model.pt` was warm-started from (needed only to literally re-run step 2's training
  command; not used at inference).
- `outputs/protocol/folds/`, `outputs/protocol/geo_cells_multi_full/` — the fixed CV folds and
  the full-data geo-cell partition `final_model.pt` was trained against (deterministic,
  rebuilt automatically by `train_final.py`/`predict.py` if missing — included so this
  submission does not depend on that rebuild step matching bit-for-bit).
- `writeup/main.tex` (+ `figures/`) — the 1-2 page report source. **Not pre-compiled to PDF**
  in this environment (no LaTeX toolchain available); compile with e.g.
  `tectonic writeup/main.tex` or any standard LaTeX distribution before submitting.
