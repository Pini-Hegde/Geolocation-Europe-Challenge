# Literature Review — Image Geolocation

**Purpose of this file:** survey the actual research lineage behind "predict lat/lng from a
photo," so the next architecture decision in `Progress.md` Phase 5 / `Project_plan.md` Phase D
is grounded in what's been tried and *why* it worked, not picked off a leaderboard. Every
candidate pulled from here into `Experiment_Design.md` still has to clear the same bar as
everything else in this project: **why should it help our specific diagnosed bottleneck**
(cell-classification accuracy, currently 41.8% with EfficientNet-B0 — see `Progress.md` §T3.1),
not "because it's a known technique."

**Filter applied throughout:** most of this literature assumes web-scale data (millions of
images) and/or no param ceiling. We have **~11.7k labelled images and a 5M-param hard limit,
no external/generated data**. Every paper below is read through that lens — what's the
*idea*, and does it survive being ported to our budget, not "can we reproduce the paper."

---

## 1. Classification-over-geo-cells — the paradigm we're already using

### PlaNet (Weyand, Kostrikov, Philbin, 2016) — [arXiv:1602.05314](https://arxiv.org/abs/1602.05314)
The foundational paper for this whole family. Frames geolocation as classification, not
regression: subdivide the Earth's surface into thousands of cells (adaptive-size, via Google's
S2 geometry library — denser cells where photo density is higher) and train a CNN to predict
the cell. Key insight we already independently arrived at in `src/protocol/geo_cells.py`:
**adaptive cell density beats a fixed grid** — PlaNet's S2 cells and our per-country
KD-tree-median-split (MIN=40/MAX=120 images) are the same idea, just a different partitioning
algorithm. PlaNet trained on ~126M web photos; at our scale, cell count has to be far smaller
(our 96 cells vs. PlaNet's ~26k) — we're already respecting this by keeping cells coarse
enough to have ~100 images each.

### CPlaNet (Seo, Weyand, Sim, Han, ECCV 2018) — [arXiv:1808.02130](https://arxiv.org/abs/1808.02130)
Directly names the tension we're living with: **fewer/larger cells → lower location precision;
more/smaller cells → fewer images per class → overfitting**, and a bigger output layer.
Their fix: **combinatorial partitioning** — build several *independent, coarse* partitionings
of the map (e.g. 3 different grids, each independently coarse enough to have plenty of data
per cell), then take the *intersection* of which coarse-cell-combination a training point
falls into as a much finer effective label — without needing enough images to directly
populate that many fine cells. Effective resolution is multiplied, not divided, per unit of
required training data. **Relevant to us**: our failure mode (cell accuracy 41.8%, correct-cell
offset error already smallish per the rolled-back attempt's error analysis) is structurally
the CPlaNet tradeoff — see §5 candidate below.

### Individual Scene Networks / ISNs (Müller-Budack, Pustu-Iren, Ewerth, ECCV 2018) — [PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf), [code](https://github.com/TIBHannover/GeoEstimation)
Two ideas, both directly relevant:
1. **Hierarchical multi-partitioning**: classify at *multiple* granularities simultaneously
   (coarse/medium/fine cell hierarchies) with a shared backbone and multiple heads, combining
   predictions across scales rather than committing to one grid resolution. This is a superset
   of what we're doing (we only have two levels: country, then cell) — a third intermediate
   level (e.g. country → region → cell) could give the offset head a better-conditioned prior
   even when the finest-level cell prediction is wrong.
2. **Scene-aware branching**: train separate sub-networks (or heads) for indoor/natural/urban
   scene types, because the visual cues that localize a photo differ by scene category (a
   building facade localizes differently than a landscape). **Doesn't port cleanly**: this
   needs scene labels, which we don't have and can't add without violating the "no external
   data" constraint (Places365-style scene classifiers are themselves trained on external
   data). Noted but likely not actionable — see Constraints note in `Experiment_Design.md`.

### Query-based hierarchical geolocation / "GeoDecoder" (Clark et al., 2023) — [arXiv:2303.04249](https://arxiv.org/abs/2303.04249)
Names a specific failure mode of our own architecture directly: most hierarchical
geolocation models (ours included) compute **one shared feature vector** (global-average-pooled
from the backbone) and feed it to every head — country, cell, offset all read off the same
pooled representation. This paper's argument: the visual cues that place a photo's *country*
(vegetation, terrain, light quality, road markings) are not the same cues that place it
precisely *within* a country (signage text, architectural detail, plate style) — averaging the
whole spatial feature map into one vector before branching forces every head to share a
representation tuned for none of them specifically.

**Full architecture (read directly from the paper, 2026-08-19), for accuracy before we borrow
from it:**
- **Backbone**: Swin Transformer, ImageNet-21k pretrained (beat ViT in their ablation, +0.5%
  @1km) — tens of millions of params, trained on 4.72M images. Not portable to us as-is.
- **Core mechanism (the part we *do* want)**: keep the backbone's spatial patch tokens (don't
  GAP them away) and run **learned query vectors** through a transformer decoder that
  cross-attends over those tokens: `CA(Q,K) = softmax(QK^T/sqrt(d_k))·V`. Each query pulls out
  whichever spatial regions are relevant to *that query specifically*, instead of every head
  sharing one averaged vector.
- **Two-stage decoder**: early layers are shared across all queries (self-attn + cross-attn +
  a shared FFN, general visual features extracted once); later layers specialize — self-
  attention restricted *within* each hierarchy level, with **separate FFNs per level**, so a
  query becomes "the country-level representation" vs. "the fine-cell representation."
  Ablation: 2 specialized layers on top of shared layers beat 0 specialized layers by +0.6%
  @1km (YFCC26k) — the specialization stage matters on its own, not just attention itself.
  8 total layers was their optimum (tested 3/5/8/10, diminishing returns past 8).
- **7 hierarchy levels** (684 -> 21,673 classes, S2-cell based — same coarse-to-fine adaptive
  density idea as PlaNet/our own `geo_cells.py`), each with its own query + specialized
  decoder output. 3->7 levels gained +1.5% @1km (Im2GPS3k) — more granularity helps, with
  diminishing returns.
- **Scene queries**: 16 scene categories (Places2-derived, coarsened from 365) x 7 hierarchies
  = 112 total queries — a "forest, hierarchy-4" query differs from a "city street, hierarchy-4"
  query. Needs an externally-pretrained Places2 scene classifier.
- **Loss**: sum of per-hierarchy-level cross-entropy + scene cross-entropy.
- **Inference**: multiplies per-level class probabilities top-down (independently-trained
  per-level classifiers combined post-hoc), unlike our training-time hard masking.
- TenCrop (9-crop) test-time augmentation used at eval.

**What we actually port, and what we don't** (full reasoning in §5's table):
- **Port**: learned-query cross-attention over spatial tokens instead of GAP (**T3.6**) — the
  load-bearing idea, scale-independent; and the shared-then-specialized staging principle,
  scaled way down (their 6 shared + 2 specialized layers -> our ~1 shared + 1 specialized,
  given a 5M-param whole-model budget vs. their Swin-scale one).
- **Don't port**: the Swin backbone (too big), the 7-level S2 hierarchy (already queued,
  evidence-gated, as T3.5 — their +1.5% from 3->7 levels supports trying a 3rd level
  *eventually*, not now), scene queries (external-data gray area, same call as ISNs §1),
  multiplicative post-hoc probability combination (our teacher-forced hard masking is more
  sample-efficient at our scale — 12 countries x 8 cells, ~11.7k images vs. their 4.72M).
- **Free confirmation**: TenCrop-style TTA validates our already-planned **T5.2**.
- **Implementation catch, flagged before building it**: a naive multi-head attention block at
  EfficientNet-B0's native feature width (1280-dim) costs `4 x 1280^2` ~= 6.5M params in the
  Q/K/V/O projections alone — over budget by itself. Fix: project spatial tokens down to a
  small attention dimension (e.g. 128) via a cheap 1x1 conv first, attend in that reduced
  space, project back up for the head — keeps the addition in the ~150-300K range. Verify via
  `src/protocol/param_check.py` before training, per our own gate.

### TransLocator (ECCV 2022) — [arXiv:2204.13861](https://arxiv.org/abs/2204.13861)
Dual-branch ViT: one branch on the raw RGB image, one on its **semantic segmentation map**,
cross-attending between them, with a joint scene-recognition auxiliary task. The segmentation
branch is explicitly there to be robust to appearance nuisance variation (weather, time of
day, season) that the RGB branch is sensitive to. Interesting but **two blockers for us**: (a)
needs a segmentation model to produce the maps — arguably a "tool," not "training data," so
possibly permissible, but it's a genuinely gray area against our own data-source open question
in `Experiment_Design.md` §6 (worth an explicit question if pursued); (b) transformer-based,
data-hungry, and a second full branch is expensive against a 5M budget shared with a
classification+regression backbone. **Not recommended** given cost/ambiguity vs. our simpler
options below.

---

## 2. Addressing cell-boundary discretization error — most directly applicable finding

### PIGEON / PIGEOTTO (Haas et al., 2023) — [arXiv:2307.05845](https://arxiv.org/abs/2307.05845)
Large CLIP-based system (way outside our param/data budget as a whole), but one component
transplants cleanly and cheaply: **haversine-smoothed soft labels for geo-cell
classification**. Instead of one-hot cross-entropy against the single true cell, the target
distribution over cells is `softmax(-haversine_dist(cell_centroid, true_lat_lng) / tau)` —
every cell gets partial credit proportional to its geographic proximity to the true point, not
just the winning cell. Their stated motivation matches our exact situation: **hard cell
boundaries are arbitrary discretization artifacts** — a point 5m on the wrong side of a KD-tree
split boundary is treated by plain cross-entropy as being just as wrong as a point in the
farthest cell in the country, which is false and penalizes the model for something not
learnable from the image (a point near a boundary genuinely looks like both neighboring
cells). **This is the single highest-priority candidate from this review** — see §5.

---

## 3. Continuous / retrieval-based paradigm (reference only — not directly adoptable)

### GeoCLIP (NeurIPS 2023) — [arXiv:2309.16020](https://arxiv.org/abs/2309.16020)
Contrastive image↔GPS-coordinate alignment (CLIP-style), with GPS coordinates encoded via
**random Fourier features at multiple resolutions** ("hierarchical" positional encoding for a
continuous location, not a discretized one), then retrieval via nearest-neighbor over a
location gallery at inference. Trained on 4.7M images; the image encoder is a full CLIP ViT
(hundreds of millions of params). **Not adoptable as a whole system** — both the data scale
and the CLIP encoder are far outside our constraints, and retrieval-based inference needs a
location gallery that functions like an external database. **One transplantable micro-idea**:
Fourier-feature encoding of a *cell centroid* (instead of raw lat/lng) as the regression target
for the offset head could give smoother gradients than the current linear offset — a small,
cheap thing to try if the offset head is ever specifically implicated by error analysis, but
not a priority right now since cell classification, not offset precision, is our diagnosed
bottleneck.

---

## 4. Continuous-output alternatives to hard classification

### Mixture Density Networks for geolocation (Kejriwal et al. 2017, text-geolocation domain) — [arXiv:1708.04358](https://arxiv.org/abs/1708.04358); CNN-MDN variant — [arXiv:1705.02750](https://arxiv.org/abs/1705.02750)
Predict a **mixture of Gaussians over location** instead of a single point (or a single cell +
offset). Motivation: a photo can be genuinely ambiguous between two visually-similar but
geographically distant places (their example: content plausible in both NYC and LA gets
averaged by plain MSE into a useless midpoint in Kansas). This is already on our roadmap as
**T4.1 (optional stretch)** in `Experiment_Design.md` — this review doesn't change its
priority (still stretch-tier, pursued only if a specific multimodal-confusion failure mode
shows up in error analysis), but confirms it's a recognized technique for exactly the failure
mode it's meant to catch, not a speculative addition.

---

## 5. What this means for our next Phase D candidate(s)

Ranked by **(a) directness of fit to our diagnosed bottleneck** — EfficientNet-B0 hierarchical
model's cell accuracy is 41.8%, the clear ceiling on the current 193.0km result — and
**(b) cost against the "complexity must earn its place" principle**:

| Candidate | Source | Effort | Kind | Why it targets our actual bottleneck |
|---|---|---|---|---|
| **T3.3 — Haversine-smoothed soft cell labels** | PIGEON (§2) | **Low** — swap cell cross-entropy for a soft target distribution over the same 96 cells; no new params, no architecture change | Loss-function | **TRIED 2026-08-19, close to neutral in isolation (+8.8km vs. baseline, one fold), not clearly harmful.** See outcome note below. |
| **T3.6 — Per-head query attention pooling** | Clark et al. 2023 (§1) | **Low-medium** — one small cross-attention pool per head (country vs. cell/offset) replacing shared GAP; est. +150-300K params, well inside the ~850K headroom under 5M | **Architecture** | **TRIED 2026-08-19, REGRESSED — worse than baseline by 38.6km, and the only condition that also hurt country accuracy.** See outcome note below; this candidate did not survive contact with our actual data scale. |
| **T3.4 — Combinatorial/multi-partition cells** | CPlaNet (§1) | **Medium** — 2-3 independent per-country partitionings (different KD-tree seeds/split axes), intersect for a finer effective cell grid, or ensemble their softmax outputs | Architecture (data/label structure) | **SUCCESS, 2026-08-19 — new best result: 182.9 ± 7.0km**, beating T3.1 (193.0 ± 4.2km) on every fold. Confirmed CPlaNet's mechanism directly: cell/country accuracy stayed flat vs. T3.1, all the gain came from blending two independently-imperfect coarse picks into a better point. Full numbers: `Experiment_Design.md` Tier 3, code: `src/protocol/geo_cells_multi.py` + `src/models/multi_partition_geo_classifier.py` + `src/train_multi_partition.py`. |
| **T3.5 — three-level hierarchy (country → region → cell)** | ISNs (§1) | **Medium-high** — one more classification head + one more masking stage | Architecture | Could reduce catastrophic errors from a wrong finest-cell pick within a right region, but adds a full extra head/stage — open decision (2026-08-19) whether to try this next or push T3.4's K from 2 to 3 partitions first, given T3.4 just proved out cheaply |
| Scene-branch, segmentation dual-branch | ISNs, TransLocator (§1) | High, and data-source gray areas | Architecture | **Not recommended** — needs labels/tools we don't have without touching the external-data boundary; cost/ambiguity doesn't clear the bar right now |
| Fourier-feature offset target | GeoCLIP (§3) | Low, but currently untargeted | Architecture (target encoding) | **Deferred** — offset regression isn't the bottleneck (cell accuracy is); revisit only if error analysis later implicates the offset head specifically |
| Mixture density output | MDN papers (§4) | Medium | Architecture | Already tracked as **T4.1**, unchanged priority — stretch tier, evidence-gated |

**Outcome (2026-08-19): T3.3+T3.6 tried together, regressed, reverted.** Combined full 5-fold:
210.4 ± 7.8km vs. baseline's 193.0 ± 4.2km. Isolated on fold-0: T3.6-only (230.3km) was worse
than even the combined run and the only condition where country accuracy also dropped, clearly
implicating it as the primary cause — likely because Clark et al. 2023's mechanism was
validated with 6 shared + 2 specialized transformer layers on 4.72M images, and this project's
necessarily much smaller port (1 shared layer + 2 specialized FFNs, ~9.4k images/fold) didn't
have enough signal for randomly-initialized attention to beat a simple pretrained-GAP-then-
linear-probe. T3.3-only (200.5km) was much closer to neutral. Full numbers, reasoning, and the
archived code: `failed_experiments/T3.3_T3.6_soft_labels_attention_pool/README.md` and
`Experiment_Design.md` Tier 3.

**Outcome (2026-08-19): T3.4 tried next, SUCCEEDED — new best result.** Full 5-fold: 182.9 ±
7.0km, beating T3.1 (193.0 ± 4.2km) on every single fold (not one lucky fold). Cell accuracy
(42.1%) and country accuracy (72.7%) both landed essentially unchanged from T3.1, confirming
the entire gain came from CPlaNet's actual mechanism — blending two independently-imperfect
coarse cell picks into a finer point — not from the model getting better at classification.
Succeeded where T3.3/T3.6 failed because it changes the cell-label structure, not the model:
no new randomly-initialized module competing for gradient signal against ~9.4k images/fold,
just K small linear heads on the same pretrained backbone. This run used the minimum K=2 to
test the mechanism cheaply first (per "complexity must earn its place") — open decision as of
2026-08-19 whether to push K to 3 next or move to T3.5.

---

## Sources
- [Where We Are and What We're Looking At: Query Based Worldwide Image Geo-localization Using Hierarchies and Scenes (Clark et al., 2023)](https://arxiv.org/abs/2303.04249)
- [PlaNet — Photo Geolocation with CNNs (Weyand et al., 2016)](https://arxiv.org/abs/1602.05314)
- [CPlaNet — Combinatorial Partitioning of Maps (Seo et al., ECCV 2018)](https://arxiv.org/abs/1808.02130)
- [Geolocation Estimation Using a Hierarchical Model and Scene Classification / ISNs (Müller-Budack et al., ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf) ([code](https://github.com/TIBHannover/GeoEstimation))
- [Where in the World Is This Image? / TransLocator (Pramanick et al., ECCV 2022)](https://arxiv.org/abs/2204.13861)
- [PIGEON: Predicting Image Geolocations (Haas et al., 2023)](https://arxiv.org/abs/2307.05845)
- [GeoCLIP: CLIP-Inspired Alignment between Locations and Images (Vivanco et al., NeurIPS 2023)](https://arxiv.org/abs/2309.16020)
- [Continuous Representation of Location using Mixture Density Networks (Kejriwal et al., 2017)](https://arxiv.org/abs/1708.04358)
- [Density Estimation for Geolocation via Convolutional Mixture Density Network (2017)](https://arxiv.org/abs/1705.02750)

*(Written 2026-08-19. Read alongside `Experiment_Design.md` — that file owns the actual
results log and constraint definitions; this file only owns "what the literature says and
why we picked what we picked.")*
