# Best result so far — 59.22 km median

**Date:** 2026-08-30 · **Branch:** `pini/uni_machine_code`

Scored the way the leaderboard specifies: **5-fold cross-validation on `train_labels.csv`,
stratified by country, metrics pooled over all out-of-fold predictions.** Split used is the
organisers' shared `folds.csv` (cached at `outputs/protocol/folds/folds_official.csv`), *not*
the repo's older group-stratified `folds.json`.

---

## 1. Headline

| | median | mean | p25 | p75 | <25 km | <200 km | <750 km |
|---|---:|---:|---:|---:|---:|---:|---:|
| previous submission (leaderboard #5) | 139.8 | 438.5 | — | — | — | 0.554 | 0.802 |
| this model, raw top-1 retrieval | 108.71 | 423.9 | 7.24 | 522.85 | 0.352 | 0.579 | 0.808 |
| **this model, final decode** | **59.22** | **410.5** | **9.85** | **489.18** | **0.390** | **0.616** | **0.813** |

Pooled over all **11,758** out-of-fold predictions.

> **Quote 59.22, never 57.75.** Tuning the decoder and reporting on the same data gives
> 57.75 km. The 59.22 figure comes from cross-fitted tuning (config chosen on 4 folds,
> applied to the held-out 5th). The 1.48 km difference is the selection bias, measured.

## 2. Parameter budget — 4,959,217 / 5,000,000

| component | params |
|---|---:|
| `convnextv2_femto.fcmae_ft_in1k` backbone | 4,848,240 |
| GeM pooling exponent `p` (learned scalar) | 1 |
| projection `Linear(384 -> 192)` | 73,920 |
| PCA-whitening mean vector (192) | 192 |
| PCA-whitening matrix (192x192) | 36,864 |
| **total** | **4,959,217** |

Headroom: 40,783 params.

## 3. Architecture

Single network. No country head, no geo-cells, no offset head, no ensemble, no TTA.

```
image 448x448
  -> convnextv2_femto.fcmae_ft_in1k  (ImageNet-pretrained, FCMAE self-supervised + in1k ft)
  -> feature map  B x 384 x 14 x 14
  -> GeM pool     g = ( mean_i ( f_i.clamp(min=1e-6) ** p ) ) ** (1/p),  p learned, init 3.0
  -> Linear(384 -> 192)
  -> L2-normalise      => 192-d descriptor
```

Inference is **retrieval**, not classification: embed the query, embed the labelled gallery,
and read the coordinate off the neighbours. There is no coordinate regression anywhere.

## 4. Training

One loss. For a batch of B images with pairwise great-circle distances `d_ij`:

```
Q = softmax_j( -d_ij / tau_km )          tau_km = 15      (geographic target)
P = softmax_j( (e_i . e_j) / temp )      temp   = 0.07    (embedding similarity)
loss = KL(Q || P) = - sum_j Q_ij log P_ij
```

Self-similarity is masked out of both. The 15 km temperature is the key departure from the
old pipeline, which used `geo_scale_km = 250` — at 250 km the target cannot distinguish
1 km from 40 km, so the embedding never learns fine-grained place identity.

**Geo-aware batch sampler.** A random batch almost never contains two co-located images
(only 24.6% of the dataset has any partner within 500 m), so the loss above would see no
positives. Each batch is built as **12 anchor points x 4 images drawn from within 25 km** of
that anchor, giving both true positives and hard same-region negatives.

| setting | value |
|---|---|
| resolution | 448 x 448 |
| epochs | 40 |
| batch | 48 (12 clusters x 4 images) |
| optimiser | AdamW, wd 0.01 |
| lr | 3e-4 head / 1e-4 backbone, OneCycle, pct_start 0.15 |
| augmentation | RandAugment `rand-m7-n2-mstd0.5`, scale (0.5, 1.0), **no horizontal flip** |
| precision | AMP fp16 |
| runtime | ~35 min per fold on one A40; 5 folds in parallel |

No flip: it mirrors driving side, sign text and lane direction, which are among the strongest
country cues in Europe.

## 5. Decoding — where 49 km of the gain came from

Raw top-1 retrieval pools to **108.71 km**. Two post-hoc steps take it to **59.22 km**,
with no retraining:

1. **PCA-whitening**, fit on the *gallery folds only* (never on the held-out fold).
   `mu` and covariance from the gallery, shrinkage toward the identity, then
   `x -> (x - mu) V diag(w ** -power)`, re-normalised. Whitening alone: 109.1 -> 70.9 km.
2. **Local-mode decode.** Take the top-32 neighbours, keep those within `r` km of the top-1,
   and take their similarity-weighted spherical mean. Adds ~11 km more.

Alpha query expansion was tested and **hurt** (+4 km); it is not used.

Cross-fitted config per fold — note how stable it is:

| fold | n | config `(shrink, dim, power, radius_km)` | raw median | final median | mean | <25 km | <200 km |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 2352 | `(0.1, 192, 0.35, 120)` | 111.07 | 62.45 | 412.4 | 0.383 | 0.614 |
| 1 | 2352 | `(0.0, 192, 0.25, 120)` | 112.38 | 57.49 | 397.6 | 0.393 | 0.621 |
| 2 | 2352 | `(0.0, 192, 0.25, 120)` | 102.60 | 58.33 | 399.5 | 0.387 | 0.618 |
| 3 | 2352 | `(0.0, 192, 0.35, 120)` | 120.73 | 59.14 | 420.9 | 0.398 | 0.612 |
| 4 | 2350 | `(0.0, 192, 0.35, 120)` | 98.49 | 60.53 | 422.1 | 0.390 | 0.614 |

Every fold picks radius 120 km and the full 192 dimensions; only the shrinkage and whitening
power wobble slightly. This is a stable operating point, not a fitted artefact.

## 6. Per-country breakdown (pooled OOF, final decode)

| country | n | raw top-1 | final | <25 km | <200 km |
|---|---:|---:|---:|---:|---:|
| Iceland | 960 | 5.8 | **11.2** | 0.726 | 0.917 |
| Norway | 1000 | 19.1 | **18.0** | 0.569 | 0.773 |
| Finland | 1000 | 32.1 | **26.1** | 0.494 | 0.814 |
| Belarus | 864 | 35.4 | **27.8** | 0.471 | 0.826 |
| Turkey | 1000 | 76.0 | **31.6** | 0.453 | 0.652 |
| United_Kingdom | 1000 | 110.2 | **59.3** | 0.356 | 0.648 |
| Sweden | 1000 | 80.8 | **59.7** | 0.371 | 0.673 |
| Spain | 1000 | 264.7 | **169.7** | 0.382 | 0.515 |
| Italy | 1000 | 296.3 | **199.2** | 0.355 | 0.501 |
| Poland | 934 | 278.3 | **239.1** | 0.244 | 0.469 |
| Germany | 1000 | 357.9 | **379.7** | 0.086 | 0.302 |
| France | 1000 | 435.5 | **489.3** | 0.190 | 0.329 |

The five sparse northern/eastern countries are essentially solved. France, Germany and
Poland hold the remaining error and are where any further work should go.

## 7. Reproducing

```bash
# 1. official folds (cached; do not regenerate - not reproducible from sklearn seed 42)
#    outputs/protocol/folds/folds_official.{csv,json}

# 2. train all five folds (parallel, ~35 min each on one A40)
for k in 0 1 2 3 4; do
  ARGS="--backbone convnextv2_femto.fcmae_ft_in1k --split official --fold $k \
        --tag off_femto448_f$k --epochs 40 --dim 192 --size 448" \
  sbatch --export=ALL slurm/poc_vpr.sbatch
done

# 3. cross-fitted decoder + pooled OOF metrics
python scripts_decode_nested.py off_femto448_f     # -> 59.22 km
python scripts_final_report.py                     # -> outputs/final_report.json
```

Artefacts: weights `outputs/poc/off_femto448_f{0..4}/best.pt`, cached embeddings
`.../emb.npz`, all numbers `outputs/final_report.json`.

## 8. Honest limitations

- **`predictions.csv` for `holdout_public/` has not been generated.** This is a
  cross-validation result on `train_labels.csv` only. The organisers' hidden-test grade is a
  separate evaluation and this number does not transfer automatically to it.
- The decoder is cross-fitted, but the *backbone*, resolution, tau and epoch count were chosen
  by looking at earlier runs on related splits. That is a mild optimism not captured by the
  1.48 km selection-bias estimate.
- No ensemble, no TTA, no reranking yet. `best-of-top-32` oracle on these same embeddings is
  **8.63 km**, so most of the remaining headroom is in candidate *ranking*, not in the
  representation.
- DINOv2-S was used as a diagnostic reference only (62.23 km pooled OOF, 22M params). It is
  not part of this model and distilling it would breach the "provided data only" rule.
