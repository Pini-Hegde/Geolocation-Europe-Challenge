# Experiment Design

## Constraints (every experiment below is designed to satisfy both, no exceptions)
1. **≤ 5,000,000 parameters** in the submitted model. Ensembles are allowed only if the
   **combined** parameter count across all sub-models stays ≤ 5M.
2. **No external or generated data.** Working assumption: this rules out ImageNet-pretrained
   weights too, since a pretrained backbone has effectively learned from external data.
   **TODO: confirm with organizers by email.** Until confirmed, everything here is
   **train-from-scratch, on `train/` only.**

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

### 1. Geographic leakage check (do this FIRST, before any model is trained)
Street-level datasets are often built by driving a route and sampling frames every few
meters, which can put near-duplicate locations in both train and validation under a naive
random split — silently inflating every validation score we'd ever report. Action: compute
pairwise haversine distance between training points (at least within-country, via a spatial
index / KD-tree to keep it cheap) and check for clusters of points within, say, <500m of each
other. If clusters exist, splitting must keep each cluster entirely in one fold (group k-fold
by location cluster, not plain stratified k-fold). This determines the actual CV strategy in
§2, so it has to happen first.

### 2. Evaluation protocol
- **Stratified k-fold CV (k=5), stratified by `country`**, used as the default protocol for
  *every* experiment — including the naive baseline — not introduced later once we "get
  serious." Given training is cheap at this model size, there's no excuse to trust a single
  noisy train/val split.
- If §1 finds location clustering, upgrade to **group-stratified k-fold** (group = location
  cluster, stratify by country) so no cluster straddles folds.
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

All models below are custom or reduced-width versions of known efficient designs, sized to
stay **under 5M params at our actual input resolution, verified per Protocol §5**, and
**trained from scratch** per the constraint assumption above. Every tier from T1 onward uses
the spherical-vector target (§3) and the k-fold protocol (§2) — architecture tier and
methodology tier are independent axes, not a single ladder.

### Tier 1 — Naive baseline (smallest learned model)
- **T1.1**: Small custom CNN (4-5 conv+pool blocks, GAP, FC → 3 outputs for the unit vector).
  Est. well under 1M params. Purpose: prove the full pipeline — data loading, CV splitting,
  spherical target conversion, training loop, error analysis tooling, `predictions.csv`
  export — works correctly end to end before optimizing anything.

### Tier 2 — Parameter-efficient backbones (still direct spherical-vector regression)
- **T2.1 MobileNetV3-Small (from scratch)**, width multiplier tuned so verified params land
  1-2.5M.
- **T2.2 ShuffleNetV2 (0.5x-1x width, from scratch)**, ~1-2M params — grouped convs + channel
  shuffle, cheap comparison point against MobileNet.
- **T2.3 Custom narrow ResNet ("ResNet-lite")**: ResNet18-style block layout, channel widths
  reduced until verified under budget. Residual connections for optimization stability at
  this depth vs. the plain CNN in T1.

### Tier 3 — Task reformulation (backbone fixed at best of Tier 2)
- **T3.1 Classification + offset regression (grid-cell hybrid)**: bin the map into cells
  (informed by the country-area analysis from B1), classification head over cells + regression
  head for the offset within the predicted cell. PlaNet-style; tends to help when precise
  regression is hard relative to dataset size.
- **T3.2 Multi-task with country auxiliary head**: shared backbone, joint loss = spherical
  regression loss + country classification loss, using the provided `country` label as free
  extra supervision.

### Tier 4 — Best backbone + best formulation, advanced variants (OPTIONAL / stretch — only pursue if Tier 1-3 error analysis specifically motivates one of these)
- **T4.1 Mixture density output**: small mixture of von Mises-Fisher (spherical analogue of a
  Gaussian) or 2D Gaussian components, for images that are genuinely visually ambiguous
  between multiple plausible locations. Take the mixture mode at inference.
- **T4.2 Tiny CNN+attention hybrid (MobileViT-XXS scale, from scratch)**: local conv bias +
  some global attention for scene-level context. Risk: attention layers are more data-hungry;
  without pretraining this may need heavier regularization to not overfit ~11.7k images.
  Treat as an experiment to try and report on honestly if it underperforms, not a guaranteed win.

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
| B0     |              |             |                    |                          |            |          |          |       |
| B1     |              |             |                    |                          |            |          |          |       |
| T1.1   |              |             |                    |                          |            |          |          |       |
