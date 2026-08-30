# Session log — 2026-08-30/31 · 139.8 km -> 59.22 km

Everything below was measured in this repo. Scripts named inline; logs in `slurm_logs/`.
**Read §1 and §9 first.**

---

## 1. The one finding that mattered

The leaderboard scores **5-fold CV on `train_labels.csv`, stratified by country**, pooled over
all out-of-fold predictions (rules panel in `Leaderboard.png`; shared `folds.csv` provided).

The repo was scoring on `outputs/protocol/folds/folds.json` — `StratifiedGroupKFold` with
**500 m location clusters as groups**. Agreement with the official folds: **20.3% = chance.**

Consequence, measured:

| | official `folds.csv` | repo `folds.json` |
|---|---:|---:|
| co-located twins (<500 m) reachable in the gallery | **20.8%** | **0.0%** |
| geometric oracle (nearest gallery point), pooled OOF | 3.29 km | 4.57 km |
| DINOv2-S top-1 retrieval, pooled OOF | **62.23 km** | **110.81 km** |

500 m is almost exactly the radius inside which two street photos are *the same place*.
Grouping at 500 m deletes precisely the band where visual matching works, while leaving the
1–25 km band (geographically close, visually unrelated) intact. The geometric oracle barely
moves; the *achievable* score halves.

`folds.json` was not a bug. `leakage_check.py` is correct and its clusters are still useful.
The error was adopting it as the **selection** metric when the scoring used a different split.

**Every pre-existing checkpoint is unusable against the official folds** — 78.7–80.4% of each
official fold's validation set was in the training set of the corresponding old checkpoint.
Re-scoring is impossible; retraining is required.

---

## 2. Result

Pooled OOF over all 11,758 images, official folds.

| | median | mean | <25 km | <200 km | <750 km |
|---|---:|---:|---:|---:|---:|
| previous submission (leaderboard #5) | 139.8 | 438.5 | — | 0.554 | 0.802 |
| this model, raw top-1 | 108.71 | 423.9 | 0.352 | 0.579 | 0.808 |
| **this model, final decode (cross-fitted)** | **59.22** | 410.5 | 0.390 | 0.616 | 0.813 |
| (same, tuned+reported on same data — do NOT quote) | 57.75 | | | | |

**4,959,217 / 5,000,000 params.** Full detail in `Best_result_so_far.md`; self-contained
reproduction in `best_result/` (verified: reran standalone, got 59.22).

Per fold — note the config is stable, every fold picks radius 120 km and dim 192:

| fold | raw | final | config (shrink, dim, power, radius) |
|---:|---:|---:|---|
| 0 | 111.07 | 62.45 | (0.1, 192, 0.35, 120) |
| 1 | 112.38 | 57.49 | (0.0, 192, 0.25, 120) |
| 2 | 102.60 | 58.33 | (0.0, 192, 0.25, 120) |
| 3 | 120.73 | 59.14 | (0.0, 192, 0.35, 120) |
| 4 | 98.49 | 60.53 | (0.0, 192, 0.35, 120) |

Per country (final):

| country | n | raw | final | <25 km | <200 km |
|---|---:|---:|---:|---:|---:|
| Iceland | 960 | 5.8 | 11.2 | 0.726 | 0.917 |
| Norway | 1000 | 19.1 | 18.0 | 0.569 | 0.773 |
| Finland | 1000 | 32.1 | 26.1 | 0.494 | 0.814 |
| Belarus | 864 | 35.4 | 27.8 | 0.471 | 0.826 |
| Turkey | 1000 | 76.0 | 31.6 | 0.453 | 0.652 |
| United_Kingdom | 1000 | 110.2 | 59.3 | 0.356 | 0.648 |
| Sweden | 1000 | 80.8 | 59.7 | 0.371 | 0.673 |
| Spain | 1000 | 264.7 | 169.7 | 0.382 | 0.515 |
| Italy | 1000 | 296.3 | 199.2 | 0.355 | 0.501 |
| Poland | 934 | 278.3 | 239.1 | 0.244 | 0.469 |
| Germany | 1000 | 357.9 | 379.7 | 0.086 | 0.302 |
| France | 1000 | 435.5 | 489.3 | 0.190 | 0.329 |

France, Germany and Poland are 2,934 images and hold essentially all the remaining error.
Note Germany and France got **worse** under the local-mode decode (357.9 -> 379.7,
435.5 -> 489.3) — when retrieval is wrong there, averaging 32 wrong neighbours is worse than
one. **A per-country or confidence-gated decode is the obvious next fix.**

---

## 3. Attribution — where the 80 km came from

Identical decode applied to both rows:

| | raw top-1 | +whitening | +whitening +mode120 |
|---|---:|---:|---:|
| femto pretrained, never trained on this data | 240.92 | 128.04 | **124.11** |
| femto after our training | 108.71 | 67.19 | **57.95** |

Neither ingredient alone works. Pretraining alone is 240.9 (worse than the old 139.8).
Decoding a pretrained model is 124.1. Training without decoding is 108.7. Both together, 58.

---

## 4. The diagnostic that pointed at the fix

Candidate oracle: is the right answer *in the shortlist* (representation) or is it there and
ranked badly (scoring)? Pooled OOF.

| | top-1 | best-of-5 | best-of-10 | best-of-32 | best-of-100 |
|---|---:|---:|---:|---:|---:|
| DINOv2-S, 22M, ineligible | 62.23 | 16.91 | 12.25 | 8.19 | 6.29 |
| **ours, 4.92M** | 109.72 | 28.46 | 16.84 | **8.63** | **5.41** |

Our 4.9M shortlist is as good as DINOv2's — better at top-100. **The representation was never
the bottleneck.** A bigger or different backbone would have bought nothing. This killed the
"we need a major architecture change" hypothesis and sent the effort to the decoder, where
whitening found 41 km in one run. `scripts_diag_candidate_oracle.py`.

---

## 5. Architecture & recipe (what actually won)

```
448x448 -> convnextv2_femto.fcmae_ft_in1k -> 384x14x14 -> GeM(learned p) -> Linear(384->192) -> L2norm
```
No country head, no geo-cells, no offset head, no multi-view crops, no ensemble, no TTA.

One loss: `KL(Q || P)`, `Q = softmax(-d_ij / 15km)`, `P = softmax(sim_ij / 0.07)`.
**tau = 15 km is the key change** — the old pipeline used `geo_scale_km = 250`, which cannot
distinguish 1 km from 40 km, so the embedding never learned place identity.

**Geo-aware batch sampler** is mandatory, not an optimisation: only 24.6% of images have any
partner within 500 m, so random batches contain no positives at all. Batches are 12 anchors x
4 images drawn from within 25 km.

40 epochs, AdamW, OneCycle 3e-4 head / 1e-4 backbone, RandAugment `rand-m7-n2-mstd0.5`,
**no horizontal flip** (mirrors driving side / sign text). ~35 min per fold on one A40.

Decode: PCA-whitening fit on the gallery folds only, then spherical weighted mean over the
top-32 neighbours within 120 km of the top-1.

---

## 6. Verification performed

- **Contamination:** all 5 folds — val/gallery overlap 0, val set matches official assignment
  exactly, |val|+|gallery| = 11,758. Clean.
- **Negative control:** shuffling the gallery coordinates gives **1756.39 km** vs 57.95 real.
  The signal is coming from image content, not from decode geometry.
- **Selection bias, decoder:** cross-fitted (tune on 4 folds, apply to 5th) = 59.22 vs
  57.75 tuned-and-reported-on-same-data. Bias = 1.48 km.
- **Selection bias, epoch:** `best.pt` was saved at the best epoch *on that fold's val set*.
  Folds 1/2/4 are converged (best ~= last); folds 0 and 3 gain 3-7 km from it.
  **Honest range is 59-62 km, not 59.22 exactly.** Fix by fixing the epoch count in advance.
- `scripts_verify.py`, `scripts_attribution.py`.

---

## 7. Negative results (do not retry)

- **Alpha query expansion**: hurts, +4 km, whitened or not.
- **Longer training beats resolution? No.** 320 px / 60 ep = 93.02; 448 px / 25 ep = 89.36.
  Resolution is the lever, not schedule length. (Contradicts the earlier T6 conclusion that
  "384 is not the lever" — that was measured on *classification*, where it holds.)
- **convnextv2_atto (3.45M)** as an ensemble partner: 116.0–127.6 per fold raw, clearly worse
  than femto. Weights in `outputs/poc/off_atto448_f*` if wanted, otherwise deletable.
- **Off-the-shelf backbone screen**, 448 px, pooled OOF, no training:
  femto 240.9 · atto 286.1 · edgenext_x_small 305.3 · efficientnet_b0 328.5 · xcit_nano 335.2 ·
  pvt_v2_b0 339.0 · mobilenetv4_conv_small 386.6 · starnet_s1 465.9 · mobilevitv2_050 517.3 ·
  repvit_m0_9 534.2 · fastvit_t8 699.8 · *(DINOv2-S 22M, ineligible: 62.2)*.
  femto is the only <5M candidate with self-supervised (FCMAE) pretraining and it wins by 45 km.

---

## 8. Rules notes

- Pretrained backbones **are** allowed; all params count, frozen included (organiser Q&A).
- **DINOv2 distillation is NOT allowed.** The Q&A permits a training-only auxiliary model to
  sit outside the param limit, but requires it be "trained from scratch using only the data
  provided" — pretrained diffusion models are explicitly banned. DINOv2 is pretrained on 142M
  external images. It was used **only as a diagnostic reference** here.
- The retrieval gallery (11,758 train images + their coordinates) is provided *data*, not
  parameters. Standard for retrieval systems, but **declare it explicitly in the writeup** — a
  grader should find a k-NN-over-training-set stated, not discovered.
- The leaderboard is participant-run and honour-system ("It is us, for us. Don't fit the
  folds."). Using retrieval across an IID fold split is a modelling choice, not fold-fitting —
  but be transparent about the near-twin structure in the writeup. It is a genuinely
  interesting finding and worth marks on its own.

---

## 9. Open items for tomorrow

**Blocking the actual submission:**
1. `predictions.csv` for `holdout_public/` has **not** been generated. 59.22 is a CV number.
2. No single all-data model exists yet. Options: (A) retrain on all 11,758 with epochs fixed
   at 40; (B) model-soup the 5 fold checkpoints (`src/model_soup.py` already written);
   (C) pick one fold model. None can be validated — that is inherent.
3. Learning-curve check not run: retrain folds on 3 folds (7,055 imgs) instead of 4 to see if
   the model is still data-hungry at 9,406. Tells you whether (A) is worth it over (B).

**Headroom (top-32 oracle is 8.63 km — the ranking is where everything left is):**
4. Confidence-gated / per-country decode. Germany and France got *worse* from local-mode
   averaging; they need a different rule.
5. Local-feature reranking of the top-32 (standard VPR second stage, reuses the same feature
   map, no extra params).
6. Geo-cells as a **reranking prior over the shortlist**, not as the predictor. This is the
   salvageable version of the geo-cell idea — cell centroids cap at ~33 km oracle, the
   candidate list caps at 5.4 km.
7. Multi-scale TTA (448 + 576).

**Housekeeping:**
8. `CLEANUP_DELETE_LIST.md` — ~3.5 GB of deletions proposed, **not executed**, awaiting a
   decision. 6 of the 94 checkpoints are untracked and unrecoverable.
9. `git pull` failed — no GitHub credentials on this machine (HTTPS remote, no helper, no SSH
   key, no `gh`). Today's work is uncommitted.
10. `.venv` and `geo_dataset` in this repo are symlinks into
    `~/projects/Geolocation-Europe-Challenge/`, added to `.gitignore`.

---

## 10. Corrections I made during the session — recorded so they don't get re-quoted

- I claimed with certainty that rank 1's 47.4 km was a hidden-holdout score, not CV.
  **Wrong** — the rules panel says 5-fold CV. The user's pushback is what caught it.
- I said existing checkpoints could be "re-scored" on the official folds. **Impossible** —
  ~80% contamination.
- My "79.9% vs 33.9% of twins straddle the split" statistic measured the wrong quantity
  (all pairs, not per-query availability). Correct figure is ~21% twin-in-gallery for both
  a single IID holdout and the official 5-fold.
- My provisional **89.4 km** came from a single 200-per-country holdout that over-weighted the
  easy countries and got a lucky draw. The honest pooled-OOF equivalent is 108.7 raw.
  **89.4 should not appear anywhere.**
