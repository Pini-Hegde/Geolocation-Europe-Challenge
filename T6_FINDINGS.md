# T6 push — findings & next steps (2026-08-27)

Effort to push the geo median from ~150 km toward the 105 km reference. This is a working
log / handoff, not a polished writeup. All experiments below are **fold-0 screens**.

---

## 1. Baseline & target

| | median km (fold-0) |
|---|---|
| T4.1c_k4 (prior best screen, no orientation) | **149.7** |
| T4.1c 5-fold honest CV | 150.6 |
| Reference score to beat (`Final_Geo_Project/literature.md`, source unknown) | **105** |

Final submitted model (`submission/`) = T5.3, all-data, K=3, 96 cells.

---

## 2. The key finding: the "information ceiling" claim is wrong

`writeup/main.tex` says FR/DE/IT/PL/ES are at a fundamental precision limit for this model
size. **Not true.** Oracle analysis (fold 0) — assume perfect geo-cell classification, predict
the cell-centroid, spherical-mean-blend the K partitions:

| cells / partition (img/cell) | 1-partition oracle | K-blend oracle |
|---|---|---|
| **96 (40-120) — current** | 94 km | **85 km** |
| 192 (25-70) | 65 km | **53 km** |
| 368 (15-45) | 42 km | **33 km** |
| (country centroid only) | — | 291 km |

The model is at ~150 km with **cell top-1 accuracy only 43 %**. The gap is **cell
granularity + classification accuracy**, both fixable. Resolution and decoder tweaks are
second-order.

Finer-cell partitions are already built for **all 5 folds**:
`outputs/protocol/geo_cells_multi_{mid=192, fine=368}/`.

---

## 3. Experiments run (fold-0)

| exp | change from baseline | result | where |
|---|---|---|---|
| **E1** | orientation correction, K=4, 96 cells, from ImageNet, 36 ep | **153.6 km** | `outputs/experiments/T6_E1_orient_k4_fold0/` |
| E2 | warm-start E1 + res384 + label-smooth τ=100 + EMA + country-blend (4 changes) | ❌ ~170, cell acc collapsed 0.42→0.35 | `failed_experiments/T6_E2_stack_res384_bolted_label_smoothing/` |
| E2b | warm-start E1 + res384 only | ❌ flat 163-169 | `failed_experiments/T6_E2b_res384_warmstart_neutral/` |
| E3, E3b | 192 cells + τ=125/50 + EMA (bundled) | ❌ killed early — **EMA bug** (see below) | deleted |
| **E3c** | 192 cells, K=3, one-hot, NO EMA, NO smoothing (clean isolation) | **~160, a wash vs E1** | `failed_experiments/T6_E3c_finecells192_onehot_wash/` |
| **E3d** | E3c + haversine label smoothing τ=65 (single variable) | **148.5 km — FIRST WIN** (−5 vs E1) | `outputs/experiments/T6_E3d_cells192_hsmooth65_fold0/` |
| **E3e** | 368 cells + smoothing τ=45, K=4, proj224, emb224, 55 ep | **137.2 km** (train-eval decode) → **132.7 km** (swept decoder) | `outputs/experiments/T6_E3e_cells368_hsmooth45_fold0/` |

### E3e — 368 cells + smoothing = 137.2 (→ 132.7 swept). Converged ~ep52 (ep53-55 regressed slightly).
- Cell top-1 acc **0.13** (368 cells) — smoothing + retrieval carry it entirely.
- **Decoder sweep** (`eval_cv_sweep.json`): best `mode_w16_l0_r200_hard_m100` = **132.7 km**.
  w16 > w12 > w8 (wants HIGH retrieval weight). r200 optimal (not 250). top_m 100 > 50.
  **Offset head does NOT help** (o0 best). TTA not yet tested.
- **PER-COUNTRY (best decoder) — France/Italy/Spain are the disaster:**
  | Iceland 52 | Belarus 85 | Finland 86 | Norway 96 | Turkey 113 | Sweden 119 | UK 130 |
  | Poland 205 | Germany 273 | **Spain 311** | **Italy 363** | **France 464** |
  FR/IT/ES = 25% of val, median ~380. If they were ~200 the overall median would be ~110.
  Running diagnostic: is it country-head misclassification or bad within-country cells?

### config that got 137/133: K=4 × 368 cells (`geo_cells_multi_fine`), cell_proj_dim 224,
  emb 224, τ=45, no EMA, orientation, from ImageNet, batch 32, 55 ep, cosine.
  `--n-partitions 4 --embedding-dim 224 --cell-proj-dim 224 --label-smooth-tau 45
   --cells-dir outputs/protocol/geo_cells_multi_fine --epochs 55`

### ENSEMBLE (2026-08-29): 4-model score-space fusion = **122.3 km fold-0**
`src/ensemble_cv.py` fuses E3e + E4(GeoCLIP) + E3d + E3h at the per-gallery-point score level
(works across different K / cell schemes / embedding dims / loc-encoder). Best decoder
`w16_r150_hard_m100` = **122.3 km** median, **mean 368** (vs 427 for E3e alone — the ensemble
mostly kills catastrophic wrong-country picks). Per-country: FR 373, IT 258, DE 243, ES 216,
PL 206 still the wall; 7 easy countries all < 120.
PROGRESSION: E1 153.6 → E3d 148.5 → E3e 137.2 (→132.7 swept) → **ensemble 122.3**.
NEXT: 3 fresh E3e-recipe seeds (clean same-arch ensemble beats this grab-bag), TTA, per-country
decoder radius, then 5-fold + full-data + predictions + writeup.

### FAILED country-fix attempts (the ~0.78 country-acc ceiling looks real):
- E3f (country-weight 2.5 + contrastive-country-blend 0.6): no-op on country acc, worse median.
- E3g (res384 + MLP country head, bundled): stalled, killed ep8.
- country-gated decode (--country-boosts / --country-topk in eval_cv): no help — France stays
  467 km even with top-2 country mask (model is *confidently* wrong, not uncertain).
- E3h (local_fraction 0.40 tighter crops): ~neutral, killed ep26.
- E4 (GeoCLIP loc encoder, shared embedding): standalone ~worse than E3e (aux loss drags the
  retrieval embedding), but architecturally diverse → useful ensemble member.

### E3e DIAGNOSTIC — country classification is the bottleneck (`_diag_e3e.py` / scratchpad)
country-head acc: France **0.585**, Spain 0.65, Poland 0.65, Italy 0.71, Germany 0.75,
Norway 0.80, Sweden/Turkey 0.81, UK 0.88, Iceland/Belarus/Finland 0.92-0.94.
Confusion: France→{Germany 23, Italy 17, Poland 10}; Italy→{Spain, Turkey}; Spain→{Italy 29, Turkey 18}.
~45% of France/Spain images land >500 km (wrong country). Retrieval top-1 country ≈ head acc
(retrieval doesn't rescue). **If FR/IT/ES hit ~0.80 country acc → overall median ~133 → ~110.**
- E3f (country-weight 2.5 + contrastive-country-blend 0.6): **NO-OP on country acc** (0.744 =
  E3e at ep5), median 27 km worse. Killed. → `failed_experiments/T6_E3f_countryweight_noop/`.
  The ~0.78 country ceiling is capacity/feature-quality, not loss weighting.
- **E3g (RUNNING from 19:53)**: E3e recipe + **MLP country head (hidden 192)** + **input 384**
  (readable signage/plates need resolution — FR/DE separation is a text problem). Two changes,
  both target country acc. K4×368 proj160 emb224 chh192 = 4.99M. batch 16, 44 ep.
  `--country-head-hidden 192 --input-size 384 --cell-proj-dim 160`
- if E3g helps country: sweep decoder, add TTA, then GeoCLIP, then ensemble, then 5-fold.
- if not: stronger backbone (convnextv2_femto / tf_efficientnetv2_b0) on the E3e recipe.

### E3d — the breakthrough (haversine label smoothing works)
- **148.5 km**, best epoch = **39 of 40 (last epoch — NOT converged, still dropping:
  151.7→149.6→148.5 over ep 37-39). Run 55-60 epochs.**
- Cell top-1 acc only **0.22** (vs E1's 0.43) — soft targets deliberately flatten it, but
  the *distribution* is well-placed, so the mode-retrieval decode lands closer. Even the raw
  cell-argmax ("hard") decode is 163 vs E1's ~172. PIGEON mechanism confirmed.
- **Decoder: r150 > r100 > r60** (tighter is WORSE — my hypothesis was wrong; the soft cell
  distribution + mode decode wants a wider radius). r250 not yet tested. Best = `mode_w8_l0_r150`.
- config: K=3 × 192 cells (`geo_cells_multi_mid`), emb 160, no proj, no EMA, orientation,
  τ=65, from ImageNet, 40 ep. `--label-smooth-tau 65`.

### What each told us
- **Orientation correction ≈ neutral on CV.** ~9 % of train+holdout images are flagged
  upside-down in `geo_dataset/orientation_flags.json` (pipeline never used it). CV can't show
  the benefit (both models train on the flipped images); the payoff, if any, is on the
  holdout. Kept — near-zero risk, cleaner training. `src/geo/orientation.py`, `--fix-orientation`.
- **Resolution 384 is not the lever** at EfficientNet-B0 scale (warm-started: flat/worse;
  matches the earlier abandoned T5.1). From-scratch 384 not tried — deprioritised.
- **Finer cells ALONE (one-hot) = a wash.** 192-way classifier caps at 34 % cell acc (vs
  43 % at 96), overfits hard (train cell-loss → 0.05). Precision gain ≈ accuracy loss.
  The oracle 53 km is unreachable without lifting effective accuracy.

### Mistakes made (don't repeat)
1. **Bundled multiple changes per screen** (E2: 4, E3: 3) → couldn't attribute failures.
   ~3 GPU-hours wasted. One change per screen.
2. **EMA bug:** `--ema-decay 0.999` + eval-on-EMA-weights → epochs 0-8 evaluated on a
   near-untrained network (country acc read 0.20 vs true 0.59). Killed 2 good experiments
   thinking they were broken. Fix: **drop EMA**, or decay ≤ 0.995, or never judge before
   epoch ~10.
3. **Label smoothing τ must match cell spacing.** 192-cell centroids are ~111 km apart
   (median NN). τ=125 → neighbour weight 0.41 (target nearly uniform → cell acc collapses).
   Rule: **τ ≈ (median NN spacing) / 3 to / 4**. For 192 cells → τ ≈ 30-40. For 368 cells
   (spacing ~55 km) → τ ≈ 15-20.
4. **Should have done the oracle analysis first** — would have skipped E1/E2/E2b.

---

## 4. Next ideas, in priority order

### A. Finer cells + haversine label smoothing, done right  ← DO THIS FIRST
- **E3d:** K=3, 192 cells (`geo_cells_multi_mid`), `--label-smooth-tau 35`, **NO EMA**,
  emb 160, orientation, from ImageNet, 40 ep, batch 32, 288 px.
- Then decoder sweep with tight radii (see B).
- If cell-acc-equivalent improves and median drops below ~145 → go to **E3e: K=4 × 368
  cells** via `--cell-proj-dim 224 --embedding-dim 224` (already fits, 4.93 M), `τ ≈ 18`.
  Oracle ceiling 33 km.
- `haversine_smoothed_cell_loss` is implemented in `src/train_retrieval_geo.py`; the
  `--label-smooth-tau` / `--cell-proj-dim` flags are wired into
  `src/train_multiview_retrieval_geo.py`.

### B. Decoder retuning for finer cells (cheap, ~5 min, no training)
- `src/eval_cv.py` is built (ensemble + TTA + full decoder grid + per-country) but
  **untested**. Run it on any checkpoint with `--radii-km 40 60 80 100 150 --tta-flip`.
- E3c/E3d trained with mode-radius 250 km — far too wide for 192+ cells; expect 60-100 km.
- Also worth: **apply the predicted offset head in the mode decode** (currently only
  `hard_decode` uses it; the retrieval/mode decode ignores it). Even a partial within-cell
  correction on top of an 85 km centroid helps.

### C. GeoCLIP-style continuous location encoder (bigger build, medium confidence)
- `src/models/location_encoder.py` is written & tested (EEP + random Fourier features +
  MLP, 198 k params; 87 k small variant). **Not yet wired into training.**
- Adds an image↔GPS alignment term + lets the decoder score a *dense synthetic grid* of
  coordinates, not just the 11.7 k training points — directly targets sparse large countries.
- Needs: a new train script/flag, InfoNCE loss vs a GPS queue, grid-decode in predict.
- Fits with K=3 @ 192 cells + emb 160 (~4.9 M). Do after A shows finer cells + smoothing work.

### D. Ensembling (reliable 5-10 %, save for the end)
- Train 2-3 seeds / the 5 CV fold models, **fuse in gallery-score space** (CPlaNet Eq. 3 —
  add per-gallery-point hierarchy+similarity scores) BEFORE mode-decode. `eval_cv.py`
  already does multi-checkpoint score-space fusion.
- The writeup's failed ensemble fused *coordinates* — wrong; fuse distributions.

### E. Smaller backbone to buy classification capacity (speculative)
- `mobilenetv4_conv_small` (~2.5 M) frees ~1.5 M params for bigger cell heads / more
  partitions / the location encoder. Bottleneck is classification capacity, so the trade
  might win. One screen, low priority.

---

## 5. Pipeline to finish once a recipe beats ~145 fold-0

1. Rebuild finer cells for all folds — **done** (`geo_cells_multi_{mid,fine}` have all 5).
2. 5-fold CV of the winning recipe → honest median.
3. `eval_cv.py` per fold for the final decoder config (+ TTA, + per-country).
4. `src/train_final.py` — retrain on all labelled data (needs finer-cell support:
   `ensure_full_data_cells` + `FULL_CELLS_DIR` currently hardcoded to the 96-cell build;
   add a `--cells-min/--cells-max` passthrough).
5. `src/predict.py` — regenerate `predictions.csv` (add `fix_orientation` to `HoldoutDataset`
   and the `--radius-km` from step 3; add score-space ensemble if using D).
6. Update `writeup/main.tex` — remove the "information ceiling" paragraph, add the oracle
   analysis + orientation finding + the finer-cell / smoothing / decoder story.

---

## 6. Environment (rebuilt this session — was missing)

- `.venv` at repo root: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
  (torch 2.13.0+cu130, timm 1.0.28). Prefix runs with `HF_HUB_OFFLINE=1`.
- Data restored (was gitignored & absent): `geo_dataset/{train_labels.csv, holdout_public/,
  orientation_flags.json}` — copied from `~/.local/share/Trash/files/Geolocation-Europe-Challenge/`.
- 1× local RTX 4000 Ada 20 GB, no SLURM. Launch:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 setsid nohup .venv/bin/python -u -m src.train_multiview_retrieval_geo ... </dev/null &`
- Screen speed: ~150-170 s/epoch @ 288 px, ~300 s @ 384 px. 40-ep screen ≈ 100 min.
- `--eval-every N` skips the expensive retrieval eval on cheap epochs (epoch 0 + last 3
  always run). `evaluate()` skips the local-embedding einsum when all `--local-weights` are 0.

## 7. Code added this session (uncommitted, branch `pini/uni_machine_code`)

| file | what |
|---|---|
| `src/geo/orientation.py` | load `orientation_flags.json`, rotate flagged images 180° |
| `src/models/location_encoder.py` | GeoCLIP-style loc encoder (built, not wired) |
| `src/eval_cv.py` | ensemble + TTA + decoder-grid + per-country CV eval (built, untested) |
| `src/models/retrieval_geo_classifier.py` | `+cell_proj_dim` (shared low-rank trunk before cell heads → fine cells fit under 5M) |
| `src/train_retrieval_geo.py` | `+haversine_smoothed_cell_loss()` (PIGEON eq 2-3) |
| `src/train_multiview_retrieval_geo.py` | `+--fix-orientation --eval-every --ema-decay --label-smooth-tau --contrastive-country-blend --cell-proj-dim --embedding-dim` |
| `src/data/dataset.py` | `GeoDataset(fix_orientation=)` |
