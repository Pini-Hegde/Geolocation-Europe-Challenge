# Experiment Design

## Constraints (every experiment below is designed to satisfy both, no exceptions)
1. **≤ 5,000,000 parameters** in the submitted model. Ensembles are allowed only if the
   **combined** parameter count across all sub-models stays ≤ 5M.
2. **No external or generated data — but pretrained weights are allowed.** Confirmed
   2026-08-17: pretrained backbones (any source, e.g. ImageNet) are permitted, as long as the
   final submitted model is ≤5M params, and **fine-tuning the pretrained model on the
   provided `train/` data is allowed.** "No external data" applies to the training data itself
   (no extra images beyond what's provided), not to weight initialization. This changes the
   default strategy: with only ~11.7k training images, fine-tuning a small pretrained backbone
   should generally outperform training the same architecture from scratch, so pretrained
   init is the default from Tier 2 onward — from-scratch is now the exception, worth doing
   only as an explicit ablation/comparison point, not the baseline assumption.
   Mixup/cutmix-as-synthetic-data question is still open (§6) — pending a short email.

Every candidate architecture in this doc gets its parameter count **verified programmatically**
before it's trained (see Protocol §5) — not eyeballed from a paper's reported number, since
reported counts often assume a different input resolution or a classifier head we won't use.

---

## Guiding principle: complexity must earn its place
This is a **graded project** (50% method/code quality, 35% writeup reasoning, 35%+50%+15%
rubric explicitly rewards "a clever, well-argued idea that slightly underperforms" over a
tuned-but-unexplained complex one) — so added complexity is a cost, not a default. Before
adding anything beyond Tier 3 (a new head, an attention block, an ensemble), we need to be
able to state in 2-3 sentences *why* it should help, based on something actually observed
(e.g. error analysis pointing at a specific failure mode) — not "because it's a known
technique." **Tier 1-3 are the core path. Tier 4-5 are optional stretch goals**, pursued only
if evidence specifically motivates them — see tier notes below.

## Part 1 — Methodology & Protocol

This section comes before architecture choice on purpose: with ~11.7k training images and a
5M-param ceiling, how we evaluate and what we ask the model to predict will matter more than
which named backbone we pick. Get this wrong and every number in the results log downstream
is untrustworthy, regardless of architecture.

### 1. Geographic leakage check (do this FIRST, before any model is trained) — **DONE, 2026-08-17**
Street-level datasets are often built by driving a route and sampling frames every few
meters, which can put near-duplicate locations in both train and validation under a naive
random split — silently inflating every validation score we'd ever report. Action: compute
pairwise haversine distance between training points (at least within-country, via a spatial
index / KD-tree to keep it cheap) and check for clusters of points within, say, <500m of each
other. If clusters exist, splitting must keep each cluster entirely in one fold (group k-fold
by location cluster, not plain stratified k-fold). This determines the actual CV strategy in
§2, so it has to happen first.

**Result:** ran `src/protocol/leakage_check.py` (haversine BallTree, nearest-neighbor distance
+ connected-components clustering at several thresholds). No exact-duplicate coordinates, and
the nearest-neighbor distance histogram is smooth with no artificial spike near zero (so this
isn't a duplicate-image artifact) — but at the 500m threshold, **24.56% of training points
(2,888 / 11,758) fall into a multi-point cluster**, up to 32 images in one cluster, and it
scales further at looser thresholds (34.8% at 1km, 46.2% at 2km). Leakage is real and
unevenly distributed by country — worst in **Iceland (47.5%)** and **Norway (45.4%)**
(plausibly sparse road networks → repeated sampling along the same route), lightest in
**Germany (7.4%)** and **France (10.2%)**. Full breakdown, histogram, and per-country table:
`outputs/protocol/leakage_check/summary.md`. Cluster assignment (filename → cluster_id,
500m threshold) saved to `outputs/protocol/leakage_check/clusters_500m.csv` for use as the
group key in group-k-fold.

**Decision: use group-stratified 5-fold CV** (group = 500m location cluster, stratify by
country) for every experiment from here on — plain stratified k-fold is ruled out.

### 2. Evaluation protocol
- **Group-stratified k-fold CV (k=5)** — group = 500m location cluster from §1's leakage
  check, stratified by `country` — used as the default protocol for *every* experiment,
  including the naive baseline, not introduced later once we "get serious." §1 found real
  clustering (24.6% of points affected at 500m), so plain stratified k-fold is ruled out;
  every cluster must land entirely in one fold. Given training is cheap at this model size,
  there's no excuse to trust a single noisy train/val split either.
- Report mean ± std of median-km across the 5 folds, not just a single fold's number — this
  tells us whether an architecture is actually better or within noise.
- Fixed, saved fold indices (e.g. a `folds.json` with image filenames per fold) so every
  experiment in the results log is compared on identical splits.

### 3. Target parameterization (a correctness fix, not a complexity tier)
Regressing raw `(lat, lng)` with MSE is naive independent of model size, for two reasons:
- **Longitude wraps at ±180°** — predicting 179° vs. a truth of -179° is 2° apart in reality
  but ~358² in squared-degree loss.
- **A degree of longitude is not a constant distance** — it shrinks by `cos(lat)` toward the
  poles, so raw-degree MSE doesn't correspond to physical (haversine) error uniformly across
  the map.

Fix (used in real geolocation research — PlaNet, ISNs): regress a **unit vector on the
sphere**, `x = cos(lat)cos(lng)`, `y = cos(lat)sin(lng)`, `z = sin(lat)`, with a cosine/angular
loss (`1 - cos(angle between pred and true vector)`), which is a smooth, differentiable proxy
for haversine distance and has no wraparound discontinuity. Convert back to `(lat, lng)` via
`asin`/`atan2` only at inference/export time. **This parameterization is used from the naive
baseline (Tier 1) onward** — it's not an "advanced" trick, it's fixing a broken target
representation.

### 4. EDA-driven baselines (not throwaway sanity checks — they answer a real question)
The point of these is to split total error into "got the right country" vs. "precision within
the country," which then tells us where to spend modeling effort:
- **B0 — Global mean predictor**: predict the training-set centroid for every image.
- **B1 — Oracle-country-centroid**: given the *true* country (cheating, upper-bound only),
  predict that country's centroid. This is a ceiling on what perfect country classification
  alone buys us — if B1's median km is already low, coarse classification matters more than
  fine regression; if it's still high (likely, since some countries like France/Germany/Poland
  span huge areas), within-country precision is where the real work is.
- **B2 — Per-country image-count-weighted random guess**: sanity floor.

### 5. Constraint-compliance check (automated, run before training any candidate)
A short script that instantiates the model at the actual input resolution we'll train at,
runs `sum(p.numel() for p in model.parameters())`, and asserts `≤ 5,000,000` — logged into
the results log as a pass/fail, not assumed from a paper. Catches surprises from input-size-
dependent FC layers before we burn a training run on something over budget.

### 6. Augmentation policy (must not cross into "synthetic/generated data")
- **Safe, standard**: random crop, color jitter, mild rotation, slight blur/noise — these
  transform provided images, they don't create new ones.
- **Judgment call, avoid unless discussed**: horizontal flip. Flipping can invert real
  geolocation cues (driving side of the road, text/signage orientation), so it may not be a
  label-preserving transform for this task specifically — flag and test its effect rather
  than assuming it's free signal like in generic image classification.
- **Disallowed**: anything that blends or generates new composite images (e.g. mixup/cutmix
  across two different training images), since that plausibly creates a "synthetic" sample
  not in the provided data. Flagging for the same organizer email as the pretrained-weights
  question.

### 7. Error analysis tooling (built once, reused for every experiment)
- Predicted-vs-true points plotted on a map, per fold.
- Country-confusion matrix (nearest-country-to-prediction vs. true country).
- Error histogram (per-image haversine km) — median/mean alone hide whether error is
  uniformly mediocre or a few catastrophic misses dragging the mean.
- Per-country breakdown of median km, to see if error is concentrated in specific
  countries (e.g. adjacent, visually-similar ones, or the underrepresented Belarus/Poland).

---

## Part 2 — Architecture tiers (naive → complex)

All models below are sized to stay **under 5M params at our actual input resolution, verified
per Protocol §5**. As of 2026-08-17, **pretrained weights are allowed** (see Constraints §2) —
Tier 2+ now defaults to ImageNet-pretrained init + fine-tuning on `train/`, with from-scratch
kept only as an explicit ablation. Every tier from T1 onward uses the spherical-vector target
(§3) and the k-fold protocol (§2) — architecture tier and methodology tier are independent
axes, not a single ladder.

### Tier 1 — Naive baseline (smallest learned model)
- **T1.1**: Small custom CNN (4-5 conv+pool blocks, GAP, FC → 3 outputs for the unit vector),
  trained from scratch. Est. well under 1M params. Purpose: prove the full pipeline — data
  loading, CV splitting, spherical target conversion, training loop, error analysis tooling,
  `predictions.csv` export — works correctly end to end before optimizing anything. Kept
  from-scratch on purpose (it's the pipeline smoke test, not a competitive candidate).

### Tier 2 — Parameter-efficient pretrained backbones, fine-tuned (still direct spherical-vector regression)
- **T2.1 MobileViT-XS, ImageNet-pretrained (`timm`, `mobilevit_xs.cvnets_in1k`)**, fine-tuned
  end-to-end — **new default primary candidate, decided 2026-08-18** (supersedes plain
  MobileNetV3-Small; see rationale below). Feature extractor ~1.93M params (verified,
  256x256 input, pooled 384-dim output via `num_classes=0`), leaving ~3M+ of budget spare
  for a regression head, an attention-pooling variant, or a Tier 3 auxiliary head. Hybrid
  CNN+transformer: conv stem for local texture/edges, transformer blocks for global
  spatial attention (~74.8% ImageNet top-1, notably stronger than MobileNetV3-Small's ~67%
  at a similar-or-smaller param count) — this is Tier 4.2's "tiny CNN+attention hybrid"
  idea, pulled forward and made the default rather than an optional stretch goal, because
  it gets the attention benefit in an already-pretrained, proven form instead of training
  attention layers from scratch on ~11.7k images (which is the specific data-hunger risk
  flagged in T4.2's original note). Rationale for not building a bespoke from-scratch
  CNN+attention model instead: attention has no locality/translation inductive bias, so it
  needs scale or pretraining to earn its keep — a hand-built version would be fighting the
  data budget rather than using a proven, already-pretrained one for free.
  - **T2.1b (ablation) attention-pooling head**: replace plain global-average-pooling with a
    single learned query cross-attending over the 7x7 (49-token) spatial feature map
    (project 576(or 384)→256, then multi-head cross-attention, ~350-450k added params) —
    cheap, isolated ablation against plain pooling on the identical backbone/data, gives a
    clean evidenced number for the writeup rather than an assumed one.
- **T2.2 MobileNetV3-Small, ImageNet-pretrained**, fine-tuned end-to-end — ~2.5M params,
  kept as a comparison point against T2.1 (weaker, pure-CNN baseline; was the original
  default primary candidate before the MobileViT-XS decision).
- **T2.3 GhostNet 1.0x, ImageNet-pretrained (`timm`)**, fine-tuned — ~5.2M params (tight,
  uses nearly the whole budget alone) — very param-efficient pure-CNN design, comparison
  point for "how much is the transformer half of T2.1 actually buying us" against a strong
  conv-only alternative at a similar accuracy class.
- **T2.4 ShuffleNetV2 (0.5x-1x width), ImageNet-pretrained**, fine-tuned — ~1-2M params,
  cheapest comparison point, leaves the most param budget spare for Tier 3 auxiliary heads.
- **T2.5 (ablation) Best of T2.1-2.4, from scratch**: same architecture, random init, same
  protocol — isolates how much pretraining is actually buying us on this task/dataset size.
  Cheap to run once the pretrained version is built (just flip the init) and gives the writeup
  a concrete, evidenced number instead of an assumed claim.

### Tier 3 — Task reformulation (backbone and formulation both open)
T2.1's full 5-fold result (459.9±19.4km) was 171.8km short of B1 (288.1km); pointed at
country/region-level confusion, not backbone precision, ruling out a same-formulation
backbone swap as the next step.

- **T3.1 Hierarchical country -> per-country-cell classifier + offset regression** —
  **DONE, 2026-08-19.** Hard-masked (not soft-gated) hierarchical head, cells built
  independently per country (96 cells/fold, 8/country, ~100 img/cell), 224px, backbone
  compared 3 ways. **Best: EfficientNet-B0, 193.0 ± 4.2km** (cell acc 41.8%, country acc
  72.8%) — beats B1 by 95.1km, current best reportable number. Full story (incl. the
  rolled-back flat-classifier/soft-gate attempt that preceded this design) in the results log
  below and `Progress.md` §T3.1. **Diagnosed bottleneck: cell-classification accuracy
  (41.8%)**, not offset precision within a correctly-picked cell — motivates T3.3/T3.4 below
  over further backbone tuning.
- **T3.2 Multi-task with country auxiliary head**: shared backbone, joint loss = spherical
  regression loss + country classification loss, using the provided `country` label as free
  extra supervision. Not yet run — largely superseded by T3.1's explicit country head, kept
  as a comparison point (implicit soft aux-loss vs. T3.1's hard-masked explicit head) if
  useful for the writeup narrative.
- **T3.3 Haversine-smoothed soft cell labels** + **T3.6 Per-head query attention pooling** —
  **tried together, 2026-08-19, REGRESSED. Reverted; archived (not deleted) in
  `failed_experiments/T3.3_T3.6_soft_labels_attention_pool/`** (full README there — project
  convention from this experiment onward is to archive failed attempts rather than delete
  them, per user decision). Combined full 5-fold: **210.4 ± 7.8km** vs. T3.1's 193.0 ± 4.2km —
  a clear regression, consistent across every fold (cell acc 36.1% vs. 41.8%, country acc
  flat at 72.8%). Isolated on fold-0 to find the culprit: T3.6-only (attention pool, hard
  labels) **230.3km** (cell acc 37.5%, country acc dropped to 70.9% — the only condition
  where country also degraded) was *worse* than the combined run, clearly the primary cause.
  T3.3-only (soft labels, plain GAP) **200.5km** was much closer to neutral (+8.8km, one fold
  only) — not proven, not clearly ruled out either. Diagnosis: T3.6's mechanism (Clark et al.
  2023) was validated with 6 shared + 2 specialized transformer layers on 4.72M images; this
  project's necessarily-smaller port (1 shared layer + 2 specialized FFNs, ~9.4k images/fold)
  likely didn't have enough signal for randomly-initialized attention to beat a simple
  pretrained-GAP-then-linear-probe — a genuine, evidenced case of complexity not earning its
  place (`Experiment_Design.md`'s own guiding principle). Code reverted to plain T3.1
  (`pooling` switch and `attention_pool.py` removed entirely, not just defaulted off). If
  revisited: T3.3 alone with a tuned `tau` (this used a single untuned guess, 50km) is the
  more promising thread, not T3.6.
- **T3.4 Combinatorial/multi-partition cells** (from `literature.md` §1, CPlaNet) — **DONE,
  2026-08-19. NEW BEST RESULT: 182.9 ± 7.0km** (full 5-fold, EfficientNet-B0), beating T3.1
  by 10.1km — **on every single fold**, not just on average (fold-by-fold: -9.3, -4.8, -14.7,
  -10.7, -10.9km). Design: K=2 independent per-country KD-tree partitions of the same points
  (`src/protocol/geo_cells_multi.py`, randomized split-axis choice via an optional `rng` on
  `geo_cells.py`'s existing builder — the canonical T3.1 partition is untouched, verified),
  same MIN=40/MAX=120 cell-size constraints as T3.1 so per-partition difficulty is unchanged.
  New model `MultiPartitionGeoClassifierRegressor`
  (`src/models/multi_partition_geo_classifier.py`, T3.1's file kept untouched on principle —
  see `failed_experiments/T3.3_T3.6_soft_labels_attention_pool/`) — K parallel country-masked
  cell heads, one shared offset head, params 4,271,434 (compliant). At eval, the K
  independently-argmaxed predicted cell centroids are combined via spherical mean
  (`blended_centroid_np`) before the offset is added. **Cell acc (42.1%) and country acc
  (72.7%) are both essentially unchanged from T3.1 (41.8%/72.8%)** — confirms the entire gain
  comes from the blending mechanism itself (two independently-imperfect coarse picks
  triangulating a better point), exactly the CPlaNet mechanism this was designed to test, not
  from the model getting better at classification. Lower risk than T3.6 by design: no new
  randomly-initialized module competing for gradient signal, just K small linear heads on the
  same pretrained backbone. Gap to the ~114km reference target: 79km (T3.1) -> **68.9km**.
  Next: could push further with K=3 (this used the minimum K=2 to test the mechanism cheaply
  first), or move to T3.5 if diminishing returns from more partitions seem likely.
- **T3.5 Three-level hierarchy (country -> region -> cell)** (from `literature.md` §1, ISNs) —
  an extra intermediate classification level between country and cell. Higher cost (one more
  head/masking stage) than T3.4 — only pursue if T3.4 doesn't close enough of the gap on its
  own.

### Tier 4 — Best backbone + best formulation, advanced variants (OPTIONAL / stretch — only pursue if Tier 1-3 error analysis specifically motivates one of these)
- **T4.1 Mixture density output**: small mixture of von Mises-Fisher (spherical analogue of a
  Gaussian) or 2D Gaussian components, for images that are genuinely visually ambiguous
  between multiple plausible locations. Take the mixture mode at inference.
- ~~**T4.2 Tiny CNN+attention hybrid (MobileViT-XXS scale)**~~ — **superseded, 2026-08-18**:
  pulled forward into T2.1 (MobileViT-XS) as the new default primary candidate rather than
  an optional stretch goal, once we confirmed a pretrained hybrid backbone was available at
  budget via `timm`. A larger, purpose-built attention hybrid (e.g. FastViT-T8, ~4M params)
  remains a possible Tier 4 stretch pick if T2.1's error analysis specifically motivates
  going bigger.

### Tier 5 — Budget-aware ensembling & free wins (OPTIONAL / stretch — T5.2 TTA is cheap and low-risk; T5.1 ensembling adds real complexity, only if it clearly beats a single model and can be explained simply)
- **T5.1 Two-model ensemble**: two independently trained models (~2-2.5M params each,
  **verified combined ≤5M**), averaged predictions (average the unit vectors, then
  renormalize, then convert to lat/lng). Diversity via different backbones or different grid
  granularities from T3.1.
- **T5.2 Test-time augmentation (TTA)**: multi-crop averaging at inference. Zero added
  parameters.

---

## Results log (fill in as experiments run)

| Exp ID | Architecture | Formulation | Params (verified) | CV median km (mean±std) | CV mean km | <200km % | <750km % | Notes |
|--------|--------------|-------------|--------------------|--------------------------|------------|----------|----------|-------|
| B0     | none (closed-form) | global spherical-mean centroid, predicted for every image | 0 | 1217.8 ± 9.1 | 1217.8 | 2.3% | 20.1% | Floor. Predicts the training-fold centroid regardless of image content. |
| B1     | none (closed-form) | oracle true-country spherical-mean centroid | 0 | 288.1 ± 4.2 | 291.1 | 28.0% | 96.9% | Cheats: uses the ground-truth country label. Ceiling on "perfect country classification alone." Gap vs. B0 (1218→288km) shows country-level signal is worth most of the error budget; B1's own 288km / 28% <200km shows real within-country precision work remains even with country solved. Worst within-country error: Norway (521km median), Spain (468km), Turkey (408km) — large/sparse countries. Best: Iceland (172km), Belarus (192km), Poland (229km) — compact countries. Motivates Tier 3 grid-cell / country-aux-head design. Full per-country table: `outputs/protocol/eda_baselines/b1_per_country.csv`. |
| ~~T1.1~~ | — | — | — | — | — | — | — | **Dropped, 2026-08-18** — see Tier 1 note above. |
| T2.1 (fold 0 screen, no reg) | MobileViT-XS, ImageNet-pretrained, `timm` | direct spherical-vector regression, `angular_loss`, 256px, backbone_lr=2e-4/head_lr=2e-3/wd=1e-4, cosine schedule | 1,934,003 | 492.8 (1 fold only) | 650.3 | 14.4% | — | **Preliminary, single-fold screen, not full CV.** First real trained model. 15ep run badly underfit (851km, confirmed via train-set eval showing no train/val gap); 60ep converged to 611.6km then plateaued (later shown to be an LR-schedule artifact, not capacity); 100ep at higher LR reached this number but revealed clear overfitting (train median 240.9km vs. val 492.8km, 251.9km gap) via the same train-set-no-aug diagnostic used for B0/B1. Superseded by the regularized run below. |
| T2.1 (fold 0 screen, regularized) | MobileViT-XS, ImageNet-pretrained, `timm` | same as above + dropout=0.3 (before head), weight_decay=5e-4 (up from 1e-4), wider augmentation (RandomResizedCrop scale 0.5-1.0, stronger ColorJitter, RandomErasing p=0.25) | 1,934,003 | 441.8 (1 fold only) | 603.1 | 16.3% | 73.8% | Single-fold screen, superseded by the full 5-fold run below. Regularization directly targeted the overfitting found above: val improved (492.8→441.8km) *and* train/val gap shrank 43% (251.9km→144.1km; train median 297.7km), confirming better generalization, not just a lucky number. Best epoch 91/100. Config: `outputs/experiments/T2.1_mobilevit_xs_fold0_screen_regularized/config.json`. |
| **T2.1 (full 5-fold CV)** | MobileViT-XS, ImageNet-pretrained, `timm` | same regularized config as above (dropout=0.3, weight_decay=5e-4, backbone_lr=2e-4, head_lr=2e-3, cosine schedule, 100 epochs) run across all 5 folds | 1,934,003 | **459.9 ± 19.4** | 618.4 | ~17.8% (avg) | ~71.3% (avg) | **Current best, real reportable T2.1 number (2026-08-18).** Run on NHR@FAU Alex cluster (`a40` partition, 1x A40, SLURM job 4041460, 2h32m). Per-fold best-val-median: fold0=476.2 (ep91), fold1=439.4 (ep88), fold2=484.4 (ep95), fold3=452.7 (ep98), fold4=446.9 (ep87) — tight spread (std 19.4km, ~4% of mean) confirms this is a stable result, not a lucky fold. Confirms the single-fold screen (441.8km) wasn't optimistic. **Still 171.8km short of B1's 288.1km oracle ceiling** — a working pretrained-backbone pipeline that doesn't yet beat "just knowing the country perfectly," motivating Tier 3 (grid-cell classification / country auxiliary head) as the next real lever rather than further tuning T2.1 in isolation. Full artifact: `outputs/experiments/T2.1_mobilevit_xs_5fold_full/cv_summary.json`. |
