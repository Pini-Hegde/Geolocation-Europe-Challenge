# Street-Level Geolocation Project: Complete Audit and Optimization Plan

**Version:** 1.0  
**Date:** 29 August 2026  
**Basis:** `CHALLENGE.md`, `CHALLENGE.pdf`, `Geolocation_Progress_Report.pdf`, and `Leaderboard.png`  
**Goal:** Reduce the hidden-test median Haversine error as far as possible while keeping the final submitted model within 5,000,000 parameters and using only the permitted data.

> **Scope boundary:** This document audits the reported architecture, experiments, evaluation logic, and research direction. The source code, checkpoints, fold files, and raw prediction files were not provided, so implementation bugs cannot yet be confirmed. A code audit should be the next step after this plan.

---

## 1. Executive decision

The current system is a serious model, not a weak baseline. It already combines multi-view image encoding, geographic classification, retrieval, several losses, and a custom decoder. The main problem is that the project has reached the point where adding more components is no longer enough. Several parts of the current design conflict with the actual metric or with each other.

The strongest route to a much lower median is:

1. **Remove hard dependence on country predictions.** Train and decode over global geographic regions so nearby locations remain nearby even when they cross a national border.
2. **Replace purely geometric cells with learnability-aware cells.** A useful cell must be geographically compact and visually distinguishable.
3. **Use a soft multi-level hierarchy rather than country -> region -> cell hard gating.** Keep multiple plausible paths alive with beam search.
4. **Separate classification and retrieval representations.** The report already shows that forcing one embedding to satisfy every objective causes negative transfer.
5. **Introduce hard-negative mining targeted at visually similar but geographically distant roads.** This directly attacks the France/Germany/Poland/Italy failure cases.
6. **Distil the diverse four-model ensemble into one compliant student.** The ensemble result proves that useful information exists which the current single model is not capturing.
7. **Calibrate and simplify inference.** Replace arbitrary score mixing and spherical averaging with cross-validated calibration, geographic mode selection, and a weighted geodesic medoid.
8. **Fix the experimental protocol.** Use out-of-fold predictions, separate decoder tuning from evaluation, and change one variable at a time.

TTA, longer training, higher resolution, or another random backbone swap may give small improvements. They are not credible explanations for a gap from 139.8 km to 47.4 km.

---

## 2. Challenge facts and non-negotiable constraints

### 2.1 Task

- Input: one 512 x 512 street-level image.
- Countries: Belarus, Finland, France, Germany, Iceland, Italy, Norway, Poland, Spain, Sweden, Turkey, and the United Kingdom.
- Training data: 11,758 labelled images with `filename`, `country`, `iso`, `lat`, and `lng`.
- Hidden test data: 2,400 images.
- Output: one latitude and longitude prediction per test image.

### 2.2 Scoring

- Main metric: **median Haversine distance in kilometres** over all 2,400 test images.
- Tie-break: **mean Haversine distance**.
- Secondary metrics:
  - percentage within 200 km;
  - percentage within 750 km.

The labels "got the city" and "got the country" are informal descriptions. They are not literal city or country accuracy measurements.

### 2.3 Restrictions

- Final submitted model: at most **5,000,000 parameters**.
- Pretrained parameters count toward this limit.
- Training must use only the data allowed by the challenge.
- The final submission is expected to contain a single compliant model, code, weights, `predictions.csv`, `requirements.txt`, `README.md`, and a one-to-two-page LaTeX report.

### 2.4 Grading consequence

- 50% method and code quality.
- 35% write-up and reasoning.
- 15% leaderboard.

This means weak evaluation practice, uncontrolled experiments, or unclear reproducibility can cost more than missing one leaderboard rank.

---

## 3. Current leaderboard gap

The following values come from the supplied leaderboard screenshot.

| Rank | Median km | Mean km | Within 200 km | Within 750 km |
|---:|---:|---:|---:|---:|
| 1 | 47.4 | 372.5 | 65.4% | 82.3% |
| 2 | 62.7 | 367.7 | 63.4% | 84.2% |
| 3 | 85.5 | 417.7 | 61.5% | 81.5% |
| 4 | 118.7 | 439.5 | 58.1% | 80.1% |
| Current model | 139.8 | 438.5 | 55.4% | 80.2% |

### 3.1 Exact improvement required

| Target | Required median | Reduction from 139.8 km |
|---|---:|---:|
| Beat rank 4 | below 118.7 km | more than 21.1 km |
| Beat rank 3 | below 85.5 km | more than 54.3 km |
| Beat rank 2 | below 62.7 km | more than 77.1 km |
| Beat rank 1 | below 47.4 km | more than 92.4 km |

### 3.2 Important correction to the progress report

Two projections in the report are incorrect:

- A result of **125-130 km would not reach rank 4** under the supplied leaderboard. It would remain behind the 118.7 km result.
- A result of **95-105 km would not reach rank 3**. It would currently be rank 4 because rank 3 requires beating 85.5 km.

Do not repeat these projections in the final report.

### 3.3 Convert percentages into image counts

There are 2,400 test images.

| System | Approximate images within 200 km |
|---|---:|
| Current: 55.4% | 1,330 |
| Rank 4: 58.1% | 1,394 |
| Rank 3: 61.5% | 1,476 |
| Rank 2: 63.4% | 1,522 |
| Rank 1: 65.4% | 1,570 |

To match rank 1 on the 200 km metric, roughly 240 additional images must move inside 200 km. However, this is not the full explanation of the median gap.

### 3.4 The main target is precision below 50-100 km

A 47.4 km median means that roughly half of the test predictions are around 47 km or better. The current model's median is 139.8 km. Therefore, rank 1 is not only moving more failures under 200 km. It is making a large portion of already reasonable predictions much more precise.

The project must start tracking:

- recall within 25 km;
- recall within 50 km;
- recall within 100 km;
- recall within 200 km;
- recall within 750 km;
- 25th, 50th, 75th, 90th, and 95th error percentiles.

The 50 km and 100 km metrics are more useful for closing the top-leaderboard gap than cell top-1 accuracy.

### 3.5 The tail also matters

The top model's mean is 372.5 km compared with the current model's 438.5 km. The top system is not only better near the median; it also reduces severe errors. Mean error is the tie-break and should be monitored separately from the median.

---

## 4. Current system summary

The progress report describes the current system as follows:

- ImageNet-pretrained EfficientNet-B0 backbone.
- Approximately 4.0M parameters in the backbone.
- Five views per image: one global view and four corner crops.
- Shared backbone and fused representation.
- One 12-way country head.
- Four independently seeded geocell classifiers.
- Each geocell belongs to one country.
- A within-cell coordinate offset head.
- A 224-dimensional normalized retrieval embedding.
- Training losses:
  - country cross-entropy;
  - Haversine-smoothed cell cross-entropy;
  - offset Huber loss;
  - geographic embedding loss;
  - view-consistency loss;
  - supervised geographic contrastive loss.
- Inference scores each training image using country probability, cell probabilities, and embedding similarity.
- The final coordinate is a density-weighted spherical mean over a selected geographic cluster.

### 4.1 Reported progression

| Stage | Reported median | Main lesson |
|---|---:|---|
| Direct coordinate regression | 450-500 km | Global regression averages incompatible modes. |
| Country + one geocell classifier | 193 km | Classification is much better than direct regression. |
| Multiple coarse partitions | 183 km | Combinatorial partitions help. |
| Multi-view retrieval and mode decode | 155 km | Retrieval adds important local information. |
| Geographic contrastive learning | 149.5 km | Representation learning helps. |
| Haversine smoothing with 192 cells | 148.5 km | Metric-aware soft labels help. |
| 368 cells + low-rank heads + longer schedule | 137.2 km | Best reported fixed single-model result on fold 0. |
| Same recipe on folds 1 and 2 | 135.3 / 138.0 km | The basic recipe is stable across three folds. |
| Four-model score fusion | 122.3 km | Diverse models contain complementary information. |

### 4.2 What should be kept, changed, or removed

| Component | Decision | Reason |
|---|---|---|
| EfficientNet-B0 | Keep as the reference backbone first | It is currently the strongest controlled baseline. |
| Orientation correction | Keep | It is supplied metadata and low risk. |
| Multi-view processing | Keep the idea, replace the crop design | Street cues are not naturally concentrated in four corners. |
| Four geocell partitions | Keep as a baseline, then compare with hierarchy | The approach has already helped, but its current country dependence is harmful. |
| Country head | Keep only as an auxiliary signal | It should not gate geographic probability. |
| Per-country Haversine smoothing | Replace | It creates artificial discontinuities at national borders. |
| Offset Huber head | Remove initially | It has not demonstrated value and consumes parameters and gradients. |
| Shared 224-d embedding for several objectives | Split into separate projections | The report shows negative transfer. |
| Five-view consistency loss | Remove or redesign | Different crops legitimately contain different evidence. |
| Geographic contrastive learning | Keep, but add hard-negative mining | Random or easy negatives do not attack the main errors. |
| Spherical mean decoder | Replace with geodesic medoid/geometric median | Spherical averaging can land between modes and is not aligned with absolute Haversine loss. |
| Horizontal-flip TTA | Treat as unproven | It changes driving side, sign placement, lane direction, and text orientation. |
| Diverse ensemble | Use as a teacher, not final inference | It is useful evidence but likely non-compliant as a final model. |

---

## 5. What the current work has established correctly

### 5.1 Direct global regression is a poor fit

A single street image can be visually compatible with several distant places. Minimizing a coordinate regression loss encourages averaging between plausible modes. The reported 450-500 km result supports moving away from plain latitude/longitude regression.

### 5.2 Geographic classification is a strong base

The jump from roughly 450-500 km to 193 km shows that geographic discretization is appropriate for this dataset.

### 5.3 Multiple partitions provide useful resolution

The CPlaNet-style idea is justified: multiple coarse partitions can create fine joint geographic resolution without requiring one enormous classifier.

### 5.4 Retrieval is valuable

The improvement from approximately 183 km to 155 km shows that gallery similarity contains information not captured by the cell heads.

### 5.5 Metric-aware label smoothing is valuable

Haversine smoothing improved the median even though hard cell top-1 accuracy decreased. This is expected: nearby-cell probability is useful even when the exact discrete cell is not top-1.

### 5.6 Diverse models are complementary

The four-model ensemble improved the fold-0 median from the mid-130s to 122.3 km and reduced the mean substantially. This is the clearest evidence that the single-model representation is leaving recoverable information unused.

### 5.7 The best single-model recipe is reasonably stable

The reported medians of 137.2, 135.3, and 138.0 km are close. The average is approximately 136.8 km. This is more reliable than relying on one unusually good fold.

---

## 6. Critical flaws in the current diagnosis and evaluation

## 6.1 Within 750 km is not literal country accuracy

A prediction can be in the wrong country but within 750 km, especially in continental Europe. A prediction can also be in the correct country but more than 750 km away in a large country.

The progress report itself reports a country-head accuracy around 0.744 for one experiment while the public within-750 metric is around 0.80. Those values are not the same quantity.

Required evaluation additions:

- true country accuracy;
- top-2 country accuracy;
- country confusion matrix;
- median error conditional on correct country;
- median error conditional on wrong country;
- high-confidence wrong-country rate;
- distance to the nearest border for cross-border errors.

## 6.2 The statement that every rank has about 82% within 750 km is false

Among the top results, the metric is roughly 80-84%, but the lower leaderboard entries shown in the screenshot are much worse. Even among the top models, 80.1% and 84.2% are not identical. Do not claim that country-level ability is the same across the whole leaderboard.

## 6.3 Cell top-1 accuracy is not the correct measure

With Haversine-smoothed targets, an adjacent cell can be geographically excellent even when it is not the exact hard label.

Replace cell top-1 accuracy with:

- distance from the true location to the highest-probability cell centroid;
- probability mass within 25, 50, 100, and 200 km;
- top-k geographic recall;
- expected Haversine distance under the predicted distribution;
- rank of the true cell;
- candidate recall before final decoding.

## 6.4 The decoder bottleneck has not been isolated properly

The report concludes that decoding is not the bottleneck after testing one offset formulation. That conclusion is too broad.

For every validation image, measure:

1. Error of the final prediction.
2. Error of the highest-scoring candidate.
3. Error of the best candidate among top 5, 10, 20, 50, and 100.
4. Error of the cell centroid.
5. Error of the retrieval nearest neighbour.
6. Error of the optimal weighted medoid over the candidate set.

Interpretation:

- If a good candidate exists but is ranked badly, the scorer is the problem.
- If candidates are good but aggregation is bad, the decoder is the problem.
- If no good candidate exists, the representation or cells are the problem.

## 6.5 Decoder sweeping may be optimistic

The 132.7 km result is described as a swept decoder result on fold 0. If retrieval weights and cluster radius were selected on fold 0 and then evaluated on fold 0, that is not an unbiased validation score.

Use one of these protocols:

- Tune on fold 0, report on folds 1 and 2.
- Tune on folds 0 and 1, report on fold 2.
- Produce out-of-fold predictions and tune the decoder using separate folds.
- Use nested cross-validation.

Do not report a decoder-tuned score as if it were a clean held-out result.

## 6.6 Several experiments change multiple variables

Examples from the report:

- E1 changes orientation handling and the number of partitions.
- E2 changes resolution, smoothing, EMA, and contrastive learning.
- E3e changes cell count, low-rank head design, training length, and smoothing temperature.
- E3f changes country-loss weight and positive-pair construction.
- E3g changes resolution and the country head.

The report says one variable at a time, but the experiment table does not consistently follow that rule.

The most important missing attribution is E3e. Its improvement cannot currently be assigned to:

- 368 cells;
- low-rank heads;
- 55 epochs;
- temperature 45 km;
- interaction among them.

## 6.7 Some conclusions are stronger than the evidence

The following statements should be weakened:

- "Country accuracy is a capacity ceiling."
- "Resolution is not useful."
- "Continuous location encoding is not useful."
- "The decoder is not the bottleneck."
- "There is no exploitable spatial structure."

A failed implementation only supports this wording:

> This specific implementation did not improve the selected validation metric.

## 6.8 The current split may not fully remove route or sequence leakage

Keeping points within 500 metres together is useful, but street-view sequences can span several kilometres. Images from one drive can share camera, weather, compression, vehicle position, and route style beyond a 500 m radius.

Required checks:

- repeat validation with 1 km, 2 km, and 5 km geographic groups;
- use perceptual hashes or local image matching to detect visually near-duplicate frames across folds;
- cluster possible sequences using image similarity plus geographic continuity;
- ensure retrieval galleries never contain validation images or same-sequence neighbours.

## 6.9 Public leaderboard feedback must not replace validation

The previous full-data model scored about 11 km better on the holdout than its CV estimate. That is one observation, not a correction factor. Do not assume every future model will beat CV by 11 km.

---

## 7. Critical architectural flaws

## 7.1 Hard country dependence conflicts with the metric

The model currently uses country probability in candidate scoring, and cells belong to exactly one country. Haversine distance does not care about political borders.

A French image 20 km from Germany should be closer to a German candidate 30 km away than to a French candidate 500 km away. The current system can do the opposite because it suppresses candidates from the predicted-wrong country.

This creates three problems:

1. Nearby cross-border cells receive no training probability.
2. Country errors become irreversible or heavily amplified.
3. Country information may be counted twice: once through the country head and again through country-specific cells.

**Required change:** use global geographic cells and global Haversine smoothing. Keep country as an auxiliary task, not the root gate.

## 7.2 Per-country smoothing creates border discontinuities

The report smooths targets only over cells inside the true country. This teaches the model that a nearby cell across a border is completely wrong while a distant cell inside the country may receive probability.

**Required change:** compute soft targets over all global cells.

## 7.3 A hard country -> region -> cell hierarchy is risky

The queued hierarchy is currently described as country -> region -> cell. If the country is wrong, the fine location cannot recover.

**Required change:** use a soft hierarchy with multiple active paths. The top-level regions may cross national borders. Country remains a parallel auxiliary label.

## 7.4 Four corner crops are not semantically matched to street scenes

Useful geolocation cues are often found in:

- lower centre: road surface and lane markings;
- upper centre: horizon, mountains, weather, and light;
- left and right sides: signs, poles, buildings, barriers, and road shoulders;
- centre distance: road geometry and traffic flow.

Four corner crops are arbitrary and can miss these areas.

**Required change:** use semantic fixed crops and learned view weighting.

## 7.5 View-consistency loss is conceptually wrong in its current form

A road crop and a sky crop should not be forced to produce the same detailed location distribution. They contain different amounts and types of evidence.

The report already shows that tighter crops became more informative but the consistency loss increased and cancelled the benefit.

**Required change:** remove symmetric consistency. Use gated fusion, evidence pooling, or teacher-student consistency only between compatible augmentations of the same view.

## 7.6 Horizontal flipping is not automatically valid

A horizontal flip changes:

- apparent driving side;
- lane placement;
- road-sign side;
- text direction;
- camera placement;
- traffic flow.

It may particularly damage United Kingdom predictions.

**Required change:** report per-country flip/no-flip results. Keep flip TTA only if it improves at least two clean folds and does not damage hard countries.

## 7.7 One embedding is serving too many objectives

The same shared representation is pushed by country classification, four cell heads, offset regression, geographic alignment, view consistency, and contrastive retrieval.

The GeoCLIP-style experiment becoming worse is evidence of gradient conflict, not proof that coordinate encoders are useless.

**Required change:** use separate low-dimensional projections:

- classification projection;
- retrieval projection;
- optional coordinate-alignment projection.

Measure gradient cosine similarity among losses before introducing any additional multi-task method.

## 7.8 The offset head consumes capacity without demonstrated value

The report says the offset head learned almost nothing and performed worse than the cell centroid or retrieval mode.

**Required change:** remove it from the next clean baseline. Revisit residual prediction only after the correct region/cell is selected reliably.

A later residual variant should predict a bounded local East-North offset relative to a prototype, not unrestricted raw latitude/longitude.

## 7.9 The current score fusion is probably uncalibrated

The decoder combines:

- log country probabilities;
- mean log probabilities from cell heads;
- cosine similarity multiplied by a fixed weight.

These signals have different scales and calibration. A cell classifier with more classes naturally produces different log-probability magnitudes than a coarse classifier. Cosine similarity is not a probability.

**Required change:** calibrate each component using clean out-of-fold predictions. Test removal of the country term before adding complexity.

## 7.10 Spherical mean is not the best estimator for absolute Haversine loss

A weighted spherical mean can fall between two distant modes, creating a location that no evidence supports. It is closer to a squared-distance estimator than an absolute-distance estimator.

**Required change:** after selecting a coherent geographic mode, use a weighted geodesic medoid or spherical geometric median.

## 7.11 Retrieval may be biased toward dense training regions

Scoring every training image can favour areas with many nearby gallery points. A dense region gets more chances to produce a strong match.

**Required change:** test local-density normalization, per-cell candidate caps, or prototype-balanced retrieval.

## 7.12 K-means must not use raw latitude/longitude Euclidean distance

The report does not state the coordinate representation used for MiniBatchKMeans. Raw longitude degrees represent different physical distances at different latitudes.

**Required change:** cluster using one of:

- 3D unit-sphere coordinates;
- an equal-area European projection;
- a local projection inside each compact region;
- a clustering method using Haversine distance.

This must be verified in the code.

## 7.13 Parameter allocation is inefficient

The backbone uses about 4.0M of the 5.0M limit, while the model also retains:

- four cell heads;
- an unused offset head;
- a 224-dimensional retrieval projection;
- several auxiliary projections.

Before replacing the backbone, remove components that do not help and measure the gain per parameter.

---

## 8. Proposed final system

The recommended system is a **global, metric-aware, hierarchical classifier-retriever with a compliant distilled student**.

```text
Input image
   |
   +-- full view
   +-- road/lower-centre view
   +-- horizon/upper-centre view
   +-- centre view
   +-- left-roadside view
   +-- right-roadside view
            |
       shared image backbone
            |
       learned view gate
            |
       fused image feature
            |
   +--------+------------------+-------------------+
   |                           |                   |
coarse global regions   middle/fine regions   retrieval projection
   |                           |                   |
soft hierarchy + consistency  global cells      hard-negative retrieval
   +---------------------------+-------------------+
                               |
                  calibrated candidate distribution
                               |
                 retrieval reranking + geographic mode
                               |
                    weighted geodesic medoid
                               |
                       predicted latitude/longitude
```

### 8.1 Backbone

Start with EfficientNet-B0 so architectural changes can be compared against the strongest established baseline. Do not change the backbone and the geographic formulation in the same experiment.

Only after the new formulation is stable, test a smaller backbone such as EdgeNeXt-XS or XXS to free parameters for heads. The exact implementation count must be measured, not copied from a paper table.

### 8.2 Semantic views

Recommended deterministic views for a 512 x 512 image:

- full image;
- lower 55-60% road view;
- upper 45-50% horizon view;
- central 60-70% view;
- left 50-60% roadside view;
- right 50-60% roadside view.

Do not assume six views are automatically better. Compare:

- full only;
- full + road + horizon;
- full + four semantic local views;
- current full + four corners.

Use a small learned gate to weight views. The gate should be able to ignore an uninformative sky or road crop.

### 8.3 Geographic hierarchy

Use three data-derived global levels:

- level 1: approximately 30-60 macro-regions;
- level 2: approximately 100-180 regions;
- level 3: approximately 300-500 fine cells.

These numbers are starting ranges, not fixed truth. Select them using cross-validation and minimum samples per class.

Important rules:

- Cells may cross country borders.
- Every fine cell has exactly one parent at each coarser level.
- Country is predicted separately as an auxiliary task.
- Inference keeps several parent paths alive using beam search.

### 8.4 Separate heads

Use:

- coarse-region classifier;
- middle-region classifier;
- fine-cell classifier or factorized partition heads;
- small auxiliary country classifier;
- 128-160 dimensional retrieval projection;
- optional separate coordinate projection only if an ablation proves value.

Remove the offset head from the first redesigned baseline.

### 8.5 Training objectives

Recommended total objective:

\[
L = \lambda_f L_{fine}
  + \lambda_m L_{middle}
  + \lambda_c L_{coarse}
  + \lambda_r L_{radius}
  + \lambda_h L_{hierarchy}
  + \lambda_{ret} L_{hard\_contrastive}
  + \lambda_{kd} L_{distill}
  + \lambda_{country} L_{country}.
\]

Do not activate every term immediately. Add them in controlled stages.

---

## 9. Core mathematical changes

## 9.1 Global Haversine-smoothed target

For true coordinate \(y\) and global cell centroid \(c_j\):

\[
q_j(y) =
\frac{\exp(-d_{hav}(y,c_j)/\tau_j)}
{\sum_k \exp(-d_{hav}(y,c_k)/\tau_k)}.
\]

Use all cells, not only cells in the ground-truth country.

### Adaptive temperature

One global temperature is unlikely to suit both dense and sparse areas. Set a local scale from neighbouring prototypes:

\[
\tau_j = \mathrm{clip}\left(
\alpha \cdot \mathrm{median}_{k \in N_K(j)} d_{hav}(c_j,c_k),
\tau_{min}, \tau_{max}
\right).
\]

This reduces over-smoothing in dense regions and under-smoothing in sparse regions.

## 9.2 Radius-mass loss

For a radius \(R\):

\[
L_R(y) = -\log \left(
\sum_{j: d_{hav}(y,c_j) \le R} p_j
+ \epsilon
\right).
\]

Use several radii, for example 25, 50, 100, and 200 km. This directly rewards putting probability close to the true location.

A weighted version:

\[
L_{radius} =
0.35L_{25} + 0.30L_{50} + 0.20L_{100} + 0.15L_{200}.
\]

The weights must be tuned on separate folds. The values above are a starting point.

## 9.3 Hierarchical consistency

Let fine-cell probabilities aggregate into middle regions and middle regions aggregate into coarse regions. Penalize disagreement between independently predicted parent probabilities and the sum of child probabilities:

\[
L_{hierarchy} =
KL\left(p_{middle} \;||\; A_{f\rightarrow m}p_{fine}\right)
+ KL\left(p_{coarse} \;||\; A_{m\rightarrow c}p_{middle}\right).
\]

This keeps the hierarchy coherent without hard gating.

## 9.4 Distillation loss

For teacher distribution \(p_T\), student distribution \(p_S\), and temperature \(T\):

\[
L_{KD} = T^2 KL\left(
\mathrm{softmax}(z_T/T)
\;||\;
\mathrm{softmax}(z_S/T)
\right).
\]

Combine it with ground-truth metric-aware targets:

\[
L = (1-\alpha)L_{truth} + \alpha L_{KD}.
\]

Use out-of-fold teacher outputs so a training sample is never taught by a teacher that memorized it.

## 9.5 Weighted geodesic medoid

For candidate coordinates \(x_i\) with weights \(w_i\), choose:

\[
\hat{x} = \arg\min_{x_i}
\sum_j w_j d_{hav}(x_i,x_j).
\]

This is simple, robust, and cannot produce an unsupported midpoint between two modes.

---

## 10. Learnability-aware geographic cells

Pure K-means optimizes geographic compactness. It does not answer whether two nearby regions look visually different enough for the model to classify.

### 10.1 Goal

Create cells that are:

- geographically compact;
- geographically connected;
- sufficiently populated;
- visually coherent;
- visually separable from neighbouring cells;
- stable across folds.

### 10.2 Minimal implementable version

1. Train the strongest clean baseline on each fold.
2. Produce out-of-fold image embeddings for every training image.
3. Convert coordinates to 3D unit-sphere coordinates.
4. Start with global geographic K-means or balanced recursive splits.
5. For each cell, calculate:
   - geographic radius;
   - number of examples;
   - within-cell visual variance;
   - confusion with adjacent cells;
   - validation median for images assigned to the cell.
6. Split a cell when:
   - it has enough examples;
   - it is geographically large;
   - its visual subgroups are separable;
   - the split improves cross-validated candidate recall.
7. Merge adjacent cells when:
   - they are repeatedly confused;
   - they are geographically close;
   - separating them does not improve validation error.

### 10.3 Research-grade graph version

Construct a graph where each training location is a node. Connect only geographically nearby points. Set edge weights using both geographic and visual similarity.

A conceptual partition cost is:

\[
J = \alpha J_{geo\_diameter}
+ \beta J_{visual\_variance}
+ \gamma J_{size\_imbalance}
+ \delta J_{cross\_validated\_confusion}.
\]

Partition the graph while preserving geographic connectivity. This is more defensible than visually clustering the entire dataset without a geographic constraint, which could place distant but similar roads in one cell.

### 10.4 Allocate cells where they are useful

Do not force equal granularity across countries or regions.

- Easy and visually distinct countries such as Iceland may need fewer fine cells.
- Large, diverse, confusing areas such as France, Germany, Italy, Spain, and Poland may need more prototypes.
- Split decisions should depend on out-of-fold error and visual diversity, not only land area.

### 10.5 Validate cells correctly

For every cell system, report:

- geographic oracle median;
- top-1 centroid median;
- top-5 candidate oracle median;
- probability mass within 50 and 100 km;
- minimum examples per cell;
- median examples per cell;
- adjacent-cell confusion rate;
- final decoded median.

A lower oracle is irrelevant if the classifier cannot place probability near the correct region.

---

## 11. Soft hierarchy and beam search

A hard hierarchy makes one early error fatal. Use a beam.

Example:

1. Keep the top 3 coarse regions.
2. Expand the top children of each coarse region.
3. Keep the top 8-12 middle paths after combining scores.
4. Expand fine cells under those paths.
5. Merge the fine-cell distribution with retrieval candidates.

Score a path with calibrated log probabilities:

\[
S(path) =
\alpha_c \log p(coarse)
+ \alpha_m \log p(middle|coarse)
+ \alpha_f \log p(fine|middle)
+ \alpha_r s_{retrieval}.
\]

The weights must be selected using cross-fitted predictions, not the same fold used for final reporting.

Country probability may be included as a weak auxiliary feature, but never as a hard mask.

---

## 12. Retrieval redesign

## 12.1 Use a separate projection

The retrieval embedding should not be forced to serve as the coordinate encoder or every classification head.

Recommended dimension: 128 or 160. Compare against 224. The lower dimension may free parameters without losing retrieval quality.

## 12.2 Build hard-negative batches

The main failure mode is visually similar rural roads that are geographically far apart. Random batches waste most negatives on easy examples.

For each anchor, sample a mixture of:

- local positives within 25-75 km, excluding the same sequence;
- local hard negatives 100-300 km away inside a similar macro-region;
- cross-border hard negatives from frequent confusion pairs;
- global visually similar negatives more than 500 km away;
- random negatives for global coverage.

Suggested batch composition as a starting point:

- 25% random/global negatives;
- 35% same-macroregion hard negatives;
- 25% confusion-pair negatives;
- 15% close positives or false-negative-aware pairs.

Do not use these exact ratios without ablation.

## 12.3 False-negative protection

Two images close in geography can be visually different, and two images from the same general area should not always be treated as strict negatives. Down-weight penalties for geographically close pairs.

A simple weighting rule can depend on both embedding similarity and Haversine distance. A more advanced option is the semivariogram-based approach in the research references.

## 12.4 Reranking

Test k-reciprocal reranking or local query expansion on the retrieval gallery.

Important limitation: this method comes from retrieval research outside geolocation. It is a hypothesis, not a guaranteed improvement. Accept it only if it improves clean folds.

## 12.5 Density correction

For candidate training image \(i\), estimate local gallery density \(\rho_i\), for example by the number of gallery points inside 50-100 km. Test:

\[
S'_i = S_i - \lambda \log(\rho_i + 1).
\]

This prevents dense areas from winning only because they have more candidate images.

---

## 13. Distil the ensemble into one final model

The four-model ensemble is currently the strongest evidence source. Weight averaging is not the same as distillation.

### 13.1 Why model soup is not enough

- The ensemble contains heterogeneous models.
- Different architectures and heads cannot be meaningfully averaged directly.
- Even same-architecture seeds may lie in incompatible weight-space basins.
- A soup may lose the diversity that produced the gain.

### 13.2 Distillation plan

1. Define one common set of global prototypes or fine cells.
2. Convert every teacher's prediction into a probability distribution over that common support.
3. Calibrate teacher temperatures on held-out folds.
4. Average teacher distributions, optionally weighted by validation quality.
5. Generate teacher outputs out-of-fold for all training images.
6. Train one student on:
   - ground-truth global Haversine targets;
   - teacher soft distributions;
   - hard-negative retrieval targets;
   - optional teacher embeddings.
7. Use teacher disagreement as uncertainty:
   - low disagreement -> stronger distillation weight;
   - high disagreement -> weaker distillation weight and broader target.

### 13.3 Specialist teachers

Train or reuse specialists for the hardest confusion groups:

- France / Germany / Poland;
- Italy / Spain / southern France;
- Nordic countries;
- United Kingdom driving-side cues;
- fine retrieval within large countries.

Their predictions can be included in the teacher distribution even though they are not submitted.

### 13.4 Rule clarification required

Before investing heavily, ask the instructors in writing whether the following are allowed:

- using multiple teacher models only during training;
- distilling them into one final model;
- storing a gallery of embeddings derived from the provided training set;
- using the provided unlabelled holdout for any transductive normalization;
- model soup as a single final checkpoint.

Do not assume permission where the wording is ambiguous.

---

## 14. Calibrated inference and uncertainty

## 14.1 Calibrate every source

Use out-of-fold predictions to temperature-scale:

- coarse head;
- middle head;
- fine head;
- country head;
- each partition head.

Standardize or calibrate retrieval similarity separately.

## 14.2 Test removing country probability first

The first decoder ablation should be:

- current decoder;
- country weight = 0;
- country weight = 0.1 of current;
- country weight = 0.25 of current;
- entropy-adaptive country weight.

The report tested increasing country influence and hard masking. It does not show a proper test of strongly reducing it.

## 14.3 Use several uncertainty signals

Useful signals:

- entropy of fine-cell distribution;
- disagreement among partitions;
- disagreement among views;
- teacher disagreement;
- retrieval-neighbour distance;
- geographic spread of top candidates;
- gap between first and second geographic modes.

Use uncertainty to blend fine and coarse predictions, not to make an irreversible country decision.

## 14.4 High-uncertainty fallback

For a highly uncertain image:

- avoid a sharp isolated nearest neighbour;
- use a broader geographic posterior;
- choose the weighted geodesic medoid of the strongest broad mode;
- optionally cap the influence of one high-similarity gallery item.

This is mainly a mean/tail improvement and must not be allowed to damage the median.

---

## 15. Controlled ablation plan

Every row below changes one main variable. Run the same seed set, fold protocol, and decoder unless the experiment is specifically a decoder experiment.

### 15.1 Reconstruct the E3e gain

| ID | Cells | Head design | Epochs | Smoothing temperature | Purpose |
|---|---:|---|---:|---:|---|
| A0 | 192 | original | original | 65 | Clean reference |
| A1 | 368 | original | original | 65 | Cell-count effect |
| A2 | 192 | low-rank | original | 65 | Head-design effect |
| A3 | 192 | original | 55 | Schedule effect |
| A4 | 192 | original | original | 45 | Temperature effect |
| A5 | 368 | low-rank | 55 | 45 | Full E3e recipe |

### 15.2 Decoder experiments

| ID | Single change | Main question |
|---|---|---|
| D0 | Reproduce current fixed decoder | Establish exact baseline |
| D1 | Country weight = 0 | Is country gating harmful? |
| D2 | Cross-fitted temperature calibration | Are score scales misaligned? |
| D3 | Weighted geodesic medoid | Is spherical mean hurting? |
| D4 | Density-normalized retrieval | Is gallery density biasing results? |
| D5 | k-reciprocal reranking | Can neighbourhood consistency improve retrieval? |
| D6 | Remove horizontal flip | Does flip damage real cues? |
| D7 | Per-country flip selection | Is flip useful only in some regions? |

### 15.3 Training-target experiments

| ID | Single change | Main question |
|---|---|---|
| T0 | Current per-country smoothing | Reference |
| T1 | Global cross-country smoothing | Does border continuity help? |
| T2 | Adaptive local temperature | Does density-aware smoothing help? |
| T3 | Add 100 km radius-mass loss | Does local probability mass improve? |
| T4 | Add 50 km radius-mass loss | Does fine precision improve? |
| T5 | Add hierarchy consistency | Does multi-level coherence help? |

### 15.4 View experiments

| ID | Views | Consistency | Fusion |
|---|---|---|---|
| V0 | Current global + corners | current | current |
| V1 | Global only | none | none |
| V2 | Global + road + horizon | none | mean |
| V3 | Global + semantic local views | none | mean |
| V4 | Same as V3 | none | learned gate |
| V5 | Same as V4 | compatible-view consistency only | learned gate |

### 15.5 Representation experiments

| ID | Single change | Main question |
|---|---|---|
| R0 | Current shared projection | Reference |
| R1 | Separate classification and retrieval projections | Does negative transfer fall? |
| R2 | Retrieval dimension 160 | Can parameters be saved? |
| R3 | Retrieval dimension 128 | Can parameters be saved further? |
| R4 | Remove offset head | Does simplification improve? |
| R5 | Hard-negative sampler | Does difficult road discrimination improve? |
| R6 | Confusion-pair sampler | Do France/Germany/Poland errors fall? |

### 15.6 Cell and hierarchy experiments

| ID | Single change | Main question |
|---|---|---|
| C0 | Current kd-tree partitions | Reference |
| C1 | Spherical global K-means | Do compact cells help? |
| C2 | Per-country K-means | Compare with C1; test border damage |
| C3 | Global hierarchy without hard country gate | Does hierarchy improve? |
| C4 | Beam search over hierarchy | Does preserving alternatives help? |
| C5 | Learnability-aware split/merge cells | Do visually separable cells improve fine recall? |

### 15.7 Distillation experiments

| ID | Single change | Main question |
|---|---|---|
| K0 | No distillation | Student reference |
| K1 | Distil coarse and fine logits | How much ensemble knowledge transfers? |
| K2 | Add confidence-weighted distillation | Does teacher disagreement handling help? |
| K3 | Add retrieval-embedding distillation | Does student retrieval improve? |
| K4 | Add specialist teachers | Do hard-country medians improve? |

---

## 16. Acceptance criteria for every experiment

A change should be promoted only when all applicable conditions are met:

1. Improves median on at least two of three clean folds.
2. Improves recall within 50 km or 100 km.
3. Does not materially worsen mean error.
4. Uses the same untouched decoder unless decoder is the variable.
5. Uses a paired bootstrap or repeated-fold estimate to show the gain is larger than ordinary noise.
6. Does not violate the parameter limit.
7. Does not add unacceptable inference complexity.
8. Has a clear causal explanation.

Suggested practical threshold:

- Treat gains below 2-3 km on one fold as noise until repeated.
- A strong result should improve both median and at least one fine-radius metric.

---

## 17. Evaluation protocol that should replace the current one

## 17.1 Split construction

- Use grouped geographic folds.
- Compare 500 m, 1 km, 2 km, and 5 km grouping.
- Add sequence or visual-similarity grouping where possible.
- Save exact split files in version control or the experiment artifact folder.

## 17.2 Cross-fitting

For each fold:

- train only on the other folds;
- build the retrieval gallery only from the training portion;
- generate validation predictions;
- save logits, embeddings, candidates, and final coordinates.

Concatenate these predictions to create a complete out-of-fold dataset for:

- calibration;
- decoder-weight fitting;
- cell redesign;
- distillation targets;
- confusion analysis.

## 17.3 Nested tuning

Separate three roles:

- model training folds;
- decoder/calibration tuning fold;
- final reporting fold.

Never optimize cluster radius, retrieval weight, or hierarchy beam on the same fold used for the headline score.

## 17.4 Required metrics

Overall:

- median and mean Haversine error;
- 25th, 50th, 75th, 90th, 95th percentiles;
- recall at 25, 50, 100, 200, 500, and 750 km;
- bootstrap confidence intervals.

Per country:

- median and mean;
- recall at 50, 100, 200, and 750 km;
- actual country accuracy;
- sample count.

Per scene/confidence slice:

- high vs low entropy;
- rural vs urban proxy if a reliable proxy can be derived from the provided data;
- near-border vs interior;
- orientation-corrected vs normal;
- flip-helped vs flip-hurt;
- dense-gallery vs sparse-gallery region.

Candidate analysis:

- top-k oracle errors;
- candidate recall within 25/50/100/200 km;
- score rank of the closest candidate;
- geographic spread of candidates.

## 17.5 Statistical comparison

Use paired bootstrap resampling of validation images:

- sample validation indices with replacement;
- calculate the median difference between two models;
- repeat, for example, 5,000-10,000 times;
- report the 95% interval.

Do not rely only on one point estimate.

---

## 18. Parameter-budget audit

### 18.1 Exact counting

Every experiment must print:

```python
n_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"parameters={n_params:,}")
print(f"trainable={trainable:,}")
assert n_params <= 5_000_000
```

Count all pretrained and frozen parameters because the rules state that pretrained weights count.

### 18.2 EfficientNet-B0 option

A plausible allocation, to be verified in code:

| Component | Approximate goal |
|---|---:|
| EfficientNet-B0 backbone | about 4.0M |
| Learned view gate | below 0.05M |
| Hierarchical low-rank heads | 0.35-0.55M |
| Retrieval projection | 0.10-0.20M |
| Country auxiliary head | below 0.02M |
| Safety margin | at least 0.10M |

Remove the offset head and reduce retrieval dimension before claiming the backbone leaves too little capacity.

### 18.3 Smaller-backbone option

EdgeNeXt research reports variants around 1.3M and 2.3M parameters. This could leave more room for geographic heads, but ImageNet accuracy does not guarantee geolocation performance.

Run it only after the new labels, views, and decoder are stable. Use the exact same data split, training schedule, head structure, and inference system for a fair comparison.

### 18.4 Training-only objects

Teacher ensembles, hard-negative indices, and OOF logits are training artifacts. They do not need to be part of final inference. Confirm the instructors' interpretation before assuming this is permitted.

---

## 19. Recommended repository structure

```text
project/
├── configs/
│   ├── baseline_e3e.yaml
│   ├── global_smoothing.yaml
│   ├── semantic_views.yaml
│   ├── hierarchy.yaml
│   └── distillation.yaml
├── data/
│   ├── build_folds.py
│   ├── sequence_audit.py
│   ├── orientation.py
│   └── datasets.py
├── geocells/
│   ├── coordinate_projection.py
│   ├── global_kmeans.py
│   ├── hierarchy_builder.py
│   ├── learnability_cells.py
│   └── cell_diagnostics.py
├── models/
│   ├── backbone.py
│   ├── semantic_views.py
│   ├── gated_fusion.py
│   ├── hierarchical_heads.py
│   ├── retrieval_head.py
│   └── student_model.py
├── losses/
│   ├── haversine_smoothing.py
│   ├── radius_mass.py
│   ├── hierarchy_consistency.py
│   ├── hard_contrastive.py
│   └── distillation.py
├── retrieval/
│   ├── gallery.py
│   ├── hard_negative_miner.py
│   ├── rerank.py
│   ├── density_correction.py
│   └── geodesic_medoid.py
├── evaluation/
│   ├── metrics.py
│   ├── bootstrap.py
│   ├── error_slices.py
│   ├── candidate_oracle.py
│   └── calibration.py
├── scripts/
│   ├── train_fold.py
│   ├── predict_fold.py
│   ├── build_oof.py
│   ├── train_teachers.py
│   ├── train_student.py
│   ├── train_full.py
│   ├── predict_holdout.py
│   └── count_parameters.py
├── outputs/
│   └── <experiment_id>/
│       ├── config.yaml
│       ├── metrics.json
│       ├── predictions.csv
│       ├── candidates.parquet
│       ├── error_by_country.csv
│       ├── environment.txt
│       └── checkpoint.pt
├── requirements.txt
└── README.md
```

### 19.1 Experiment logging requirements

Every run should save:

- experiment ID;
- git commit;
- configuration file;
- random seed;
- fold ID;
- parameter count;
- training time;
- best epoch;
- all metrics;
- prediction file;
- failure notes;
- exact command used.

This directly improves the 50% method/code portion of the grade.

---

## 20. Recommended training schedule

A staged curriculum may be more stable than enabling every loss at epoch 1.

### Stage 1: geographic classification foundation

- Train coarse, middle, and fine heads.
- Use global Haversine targets.
- Use mild image augmentation.
- No hard-negative mining yet.
- No distillation yet.

### Stage 2: retrieval specialization

- Activate the separate retrieval projection.
- Introduce regional positives and hard negatives gradually.
- Keep classification heads active.

### Stage 3: hierarchy and fine-radius optimization

- Add hierarchy consistency.
- Add one radius-mass term, starting at 100 km.
- Add 50 km only after stable training.

### Stage 4: distillation

- Load calibrated OOF teacher targets.
- Ramp distillation weight over several epochs.
- Reduce the weight when teachers disagree strongly.

### Stage 5: EMA and final fine-tuning

- Use EMA from the start of the final run if it has been validated.
- Evaluate both raw and EMA weights.
- Select based on clean CV, not training loss.

Do not add generic label smoothing on top of metric-aware Haversine targets unless a controlled ablation supports it.

---

## 21. Augmentation policy

### Keep or test

- supplied orientation correction;
- mild brightness and contrast changes;
- mild colour jitter;
- small scale changes;
- limited compression simulation if the dataset shows varying JPEG quality;
- semantic fixed crops;
- light blur only if validated.

### Avoid by default

- arbitrary rotations;
- aggressive perspective transforms;
- strong random crops that remove road context;
- random grayscale;
- heavy blur;
- unconditional horizontal flipping.

Geolocation relies on subtle colour, lane, pole, sign, and vegetation cues. Heavy augmentation can destroy the exact information the model needs.

---

## 22. Immediate execution order

Run the following in this order. Do not start with another backbone.

### Priority 1: Correct the measurement system

1. Reproduce the current fixed-decoder baseline on all three folds.
2. Add 25/50/100/200 km recalls and complete error percentiles.
3. Add actual country accuracy and confusion matrices.
4. Add top-k candidate oracle analysis.
5. Audit fold sequence leakage.

### Priority 2: Low-cost decoder corrections

6. Set country decoding weight to zero.
7. Test weighted geodesic medoid instead of spherical mean.
8. Remove horizontal-flip TTA and report per-country changes.
9. Calibrate head temperatures using cross-fitted predictions.
10. Test density normalization and reranking.

### Priority 3: Fix training-target geometry

11. Train global cross-country Haversine smoothing from scratch.
12. Test adaptive smoothing temperature.
13. Add one 100 km radius-mass loss.
14. Add 50 km radius mass only if 100 km helps.

### Priority 4: Fix visual evidence extraction

15. Replace corner crops with road/horizon/side crops.
16. Remove five-view consistency.
17. Add a learned view gate.
18. Separate retrieval and classification projections.

### Priority 5: Attack hard continental cases

19. Build confusion-driven hard-negative batches.
20. Train with false-negative-aware weighting.
21. Report France, Germany, Poland, Italy, and Spain separately.

### Priority 6: Redesign regions

22. Finish spherical global K-means.
23. Compare global K-means with per-country K-means.
24. Build a soft hierarchy.
25. Add beam search.
26. Build learnability-aware split/merge cells.

### Priority 7: Compress ensemble knowledge

27. Generate OOF teacher distributions.
28. Distil the four-model ensemble into one student.
29. Add specialist teachers if needed.
30. Train the final full-data student and submit only after it passes all compliance checks.

### Priority 8: Backbone experiment

31. Test a smaller backbone only after the above system is stable.

---

## 23. What to stop spending time on

Deprioritize the following unless new evidence appears:

- direct global latitude/longitude regression;
- increasing country-loss weight again;
- hard top-1 or top-2 country masks;
- more arbitrary resolution increases;
- additional random lightweight backbone swaps;
- more losses attached to the same shared embedding;
- offset regression in its current Huber form;
- decoder sweeps evaluated on the same fold used for tuning;
- treating horizontal-flip TTA as automatically safe;
- using cell top-1 accuracy as the main diagnostic;
- assuming longer training alone can close the leaderboard gap.

---

## 24. Failure analysis that should be added

For the 100-200 worst validation images per fold, save:

- image;
- true coordinate and country;
- predicted coordinate and country;
- error distance;
- top 10 candidate coordinates and scores;
- country probabilities;
- coarse/middle/fine probabilities;
- view-gate weights;
- retrieval neighbours;
- uncertainty signals.

Create failure groups:

- confident wrong country;
- correct country, wrong region;
- correct region, poor fine precision;
- retrieval nearest-neighbour failure;
- two-mode averaging failure;
- border case;
- low-information rural road;
- orientation issue;
- flip-induced failure;
- gallery density bias.

This will tell you which experiment to run next. Without it, the project risks guessing.

---

## 25. Risk register

| Risk | Consequence | Mitigation |
|---|---|---|
| Fold-0 decoder overfitting | Inflated CV result | Nested or cross-fitted tuning |
| Cells too fine | Low data per class and unstable training | Minimum samples, hierarchy, learnability-aware merging |
| Hard hierarchy | Irrecoverable early error | Beam search and soft parent distributions |
| Country overconfidence | Cross-border suppression | Global cells and weak auxiliary country signal |
| Loss conflict | Negative transfer | Separate projections and gradient analysis |
| Distillation copies teacher mistakes | Student inherits systematic bias | Combine teacher and truth; reduce weight on disagreement |
| Retrieval density bias | Dense regions dominate | Density normalization and candidate caps |
| Horizontal flip removes cues | Country and driving-side errors | Per-country ablation |
| Smaller backbone loses useful features | Worse median despite more head capacity | Controlled same-pipeline comparison |
| Public leaderboard overfitting | Poor final generalization | Submit only CV-validated candidates |
| Rule ambiguity | Non-compliant final system | Obtain written clarification |
| Excessive system complexity | Poor reproducibility and weak report | Stage changes, config-driven experiments, remove no-gain parts |

---

## 26. Realistic leaderboard interpretation

No honest plan can guarantee 47.4 km on an unseen test set.

Use the leaderboard thresholds as decision points:

- **Below 118.7 km:** current rank-4 threshold.
- **Below 85.5 km:** current rank-3 threshold.
- **Below 62.7 km:** current rank-2 threshold.
- **Below 47.4 km:** current rank-1 threshold.

The current plan of K-means, hierarchy, TTA, and longer training may produce useful gains, but by itself it is more likely to be an incremental strategy. The changes most capable of producing a major step are:

1. global border-aware geographic targets;
2. learnability-aware regions;
3. hard-negative representation learning;
4. soft hierarchy with beam search;
5. ensemble-to-student distillation.

Treat any expected holdout score as uncertain until it is submitted. Do not present 125-130 km as rank 4 or 95-105 km as rank 3 under the supplied leaderboard.

---

## 27. Final recommended configuration to aim for

This is the final target architecture, not the first experiment.

### Image side

- EfficientNet-B0 unless a smaller backbone wins a controlled comparison.
- Full, road, horizon, centre, left-roadside, and right-roadside views.
- Shared backbone.
- Learned view gate.
- No symmetric multi-view consistency.

### Geographic side

- Global data-derived macroregions, regions, and fine cells.
- No hard country gating.
- Country auxiliary head only.
- Global Haversine smoothing.
- Adaptive local temperature.
- Radius-mass loss at validated radii.
- Hierarchical consistency.

### Retrieval side

- Separate 128-160 dimensional projection.
- Hard negatives based on visual similarity and geographic distance.
- Confusion-pair sampling for hard continental countries.
- False-negative-aware weighting.
- Optional reranking and density correction.

### Teacher/student side

- Four-model teacher ensemble represented on one common geographic support.
- OOF teacher predictions.
- Confidence-weighted distillation.
- One final student below five million parameters.

### Inference side

- Calibrated soft hierarchy.
- Beam search.
- Candidate fusion without hard country masks.
- Geographic mode selection.
- Weighted geodesic medoid.
- Horizontal flip only if clean ablations support it.

---

## 28. Submission and report checklist

### Code package

- [ ] Training script runs from a clean environment.
- [ ] Evaluation script reproduces validation metrics.
- [ ] Holdout prediction script generates exactly 2,400 rows.
- [ ] Header is exactly `filename,pred_lat,pred_lng`.
- [ ] All filenames include `.jpg`.
- [ ] No index column.
- [ ] Model parameter count is printed and below five million.
- [ ] `requirements.txt` pins versions.
- [ ] README includes exact commands.
- [ ] Training time and hardware are documented.
- [ ] Random seeds and folds are saved.
- [ ] Final checkpoint is the exact checkpoint used for `predictions.csv`.

### One-to-two-page report

Use the limited space for:

1. one compact architecture diagram;
2. one main ablation table;
3. one error-CDF or threshold-recall plot;
4. one geographic-cell visualisation;
5. a short paragraph on failed experiments;
6. exact parameter count and training time;
7. clear distinction among:
   - public leaderboard score;
   - fixed-decoder CV score;
   - decoder-tuned score;
   - ensemble diagnostic score;
   - final compliant single-model score.

Do not fill the report with a long general introduction. The challenge explicitly rewards experiments and reasoning.

---

## 29. Code audit checklist for the next review

When the repository is available, inspect these first:

### Data and folds

- [ ] Are validation images excluded from the retrieval gallery?
- [ ] Are same-location or same-sequence images separated correctly?
- [ ] Are orientation flags applied consistently to train, validation, and holdout?
- [ ] Are latitude and longitude read in the correct order everywhere?
- [ ] Are degrees converted to radians correctly in Haversine calculations?

### Cells

- [ ] Is K-means run on projected or spherical coordinates rather than raw degrees?
- [ ] Are cell centroids computed on the sphere rather than by naive longitude averaging?
- [ ] Are empty or tiny cells handled safely?
- [ ] Are cell mappings identical between training and inference?
- [ ] Is the true target smoothed over the intended global support?

### Model

- [ ] Does the reported parameter count match the actual checkpoint?
- [ ] Are all intended parameters registered in PyTorch modules?
- [ ] Are any training-only heads accidentally retained in the final model?
- [ ] Are embeddings normalized at the correct point?
- [ ] Are view features fused in the intended order?

### Losses

- [ ] Are loss scales logged separately?
- [ ] Does one loss dominate the gradient?
- [ ] Are labels normalized after Haversine smoothing?
- [ ] Are positive and negative pairs constructed without validation leakage?
- [ ] Is the contrastive temperature stable and bounded?

### Decoder

- [ ] Are probability and similarity scales calibrated?
- [ ] Is country counted twice?
- [ ] Are all candidate coordinates in valid ranges?
- [ ] Is the cluster selection deterministic?
- [ ] Does spherical averaging fail near multiple modes?
- [ ] Are decoder hyperparameters tuned on separate data?

### Submission

- [ ] Are there exactly 2,400 predictions?
- [ ] Are there NaNs or duplicated filenames?
- [ ] Are latitude values inside [-90, 90]?
- [ ] Are longitude values inside [-180, 180]?
- [ ] Is the file order irrelevant or matched safely by filename?

---

## 30. Research basis and how it should be used

The following papers support individual design principles. Their datasets, model sizes, and external metadata are not automatically permitted in this challenge. Use the ideas, not disallowed training data.

1. **CPlaNet: Enhancing Image Geolocalization by Combinatorial Partitioning of Maps**  
   Seo et al., 2018.  
   https://arxiv.org/abs/1808.02130  
   Relevant idea: combine multiple coarse partitions to create finer effective resolution without one huge classifier.

2. **PIGEON: Predicting Image Geolocations**  
   Haas et al., 2023/2024.  
   https://arxiv.org/abs/2307.05845  
   Relevant ideas: Haversine-smoothed labels, geographic hierarchy, and location refinement. Do not copy external administrative metadata or synthetic-caption training if the challenge forbids it.

3. **GeoCLIP: CLIP-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization**  
   Cepeda et al., 2023.  
   https://arxiv.org/abs/2309.16020  
   Relevant ideas: continuous coordinate encoding, equal-earth projection, multi-scale random Fourier features, and separate image-location alignment. The current failed shared-head implementation does not disprove the general idea.

4. **OpenStreetView-5M: The Many Roads to Global Visual Geolocation**  
   Astruc et al., CVPR 2024.  
   https://arxiv.org/abs/2404.18873  
   Relevant ideas: strict spatial/sequence separation, evaluation beyond memorisation, hierarchical spatial representations, and geography-aware representation learning. The external dataset must not be used for training this challenge model.

5. **Distilling the Knowledge in a Neural Network**  
   Hinton, Vinyals, and Dean, 2015.  
   https://arxiv.org/abs/1503.02531  
   Relevant idea: transfer ensemble behaviour into one smaller deployable model through soft targets.

6. **Re-Ranking Person Re-Identification With k-Reciprocal Encoding**  
   Zhong et al., CVPR 2017.  
   https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html  
   Relevant idea: use reciprocal neighbourhood consistency to rerank a retrieval list. It must be validated for geolocation rather than assumed to work.

7. **EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications**  
   Maaz et al., 2022.  
   https://arxiv.org/abs/2206.10589  
   Relevant idea: small backbones may free parameter budget for richer geographic heads. Exact geolocation performance is unknown until tested.

8. **Enhancing Contrastive Learning for Geolocalization by Discovering Hard Negatives on Semivariograms**  
   2025.  
   https://arxiv.org/abs/2509.21573  
   Relevant idea: emphasize visually similar but geographically distant hard negatives while reducing damage from geographically close false negatives.

9. **HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation**  
   2026 preprint.  
   https://arxiv.org/abs/2601.23064  
   Relevant ideas: preserve an explicit geographic hierarchy, weight contrastive learning by Haversine distance, and keep several plausible paths through beam search. Hyperbolic geometry itself is a higher-risk optional experiment for this small-data, five-million-parameter setting.

---

## 31. Bottom line

The project is currently limited less by basic model capacity than by formulation and experimental control.

The most damaging current choices are:

- hard country dependence;
- per-country smoothing;
- arbitrary corner crops;
- forcing dissimilar views to agree;
- one embedding serving too many losses;
- uncalibrated score fusion;
- using spherical averaging for a multimodal posterior;
- treating geometric compactness as cell learnability;
- not distilling the demonstrated ensemble gain;
- tuning and reporting on the same fold.

The best sequence is:

1. fix evaluation;
2. remove country gating;
3. use global metric-aware targets;
4. replace the decoder aggregation;
5. use semantic views without harmful consistency;
6. train with hard negatives;
7. build learnability-aware global cells and a soft hierarchy;
8. distil the ensemble into one compliant student;
9. only then test a new backbone.

That sequence has the strongest chance of reducing the median substantially while also producing a defensible, reproducible final project.
