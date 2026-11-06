# Image Geolocation Challenge - Where in Europe was this photo taken?

Predicting a GPS coordinate for street-level photographs from 12 European countries, under a
5,000,000-parameter budget, using only the provided training data.

| | |
|---|---|
| **Parameter count** | **4,922,161** — 98.4% of the 5,000,000 budget |
| **Held-out score** | **34.16 km** median haversine · 359.5 km mean · 47.4% < 25 km · 65.9% < 200 km · 83.5% < 750 km |
| **Measured on** | 2,352 labelled images the scored model never saw ([How → Reproduce](#reproducing-the-3416-km-score)) |
| **Approach** | **image retrieval** against the labelled training set. No coordinate regression anywhere. |
| **End-to-end** | **≈ 74 min** from clean checkout to `predictions.csv`, one NVIDIA RTX 4000 Ada |

We measured every number in this file on the hardware named above. You can re-derive all of them
with the commands given in this file.


---

## 01. The system, end to end

The approach I used in this project is quite different from the traditional system so that is a **retrieval** system, not a regressor. It never predicts a coordinate from an image directly.


**THE TRAINING**
```
                                        40 epochs · KL soft-target loss · tau_km = 15 km
┌───────────────────────────────────────────────────────────────────────────────────────┐
│  12 anchors x 4 images from within 25 km  ───►  encoder  ───►  192-d descriptors      │
│                       ▲                                               │               │
│                       │            loss = KL( geography ‖ embedding ) │               │
│                       └──────────────  update weights  ◄──────────────┘               │
└───────────────────────────────────────────────────────────────────────────────────────┘

```

**THE INFERENCE**
```
                                              **(nothing below the encoder has parameters)

  gallery path — run once                        query path — run per photo
┌─────────────────────────────────┐            ┌─────────────────────────────────┐
│ 11,758 labelled train images    │            │ 1 unlabelled holdout photo      │
│   encoder @ 448 / 512 / 576     │            │   encoder @ 448 / 512 / 576     │
│   average, L2-normalise         │            │   average, L2-normalise         │
│                                 │            │                                 │
│  11,758 x 192        globals    │            │  192           global           │
│  11,758 x 196 x 192  tokens     │            │  196 x 192     tokens           │
│  11,758 known coordinates       │            │                                 │
└────────────────┬────────────────┘            └────────────────┬────────────────┘
                 │                                              │
                 │   fit PCA-whitening, power 0.35              │
                 │   fit CSLS hub stats, k = 10   ─────────────►│  apply both
                 │                                              │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                    cosine similarity,  1 query  vs  11,758 gallery
                                         │   minus  beta x hub,   beta = 0.25
                                         ▼
                                 top-64 shortlist
                                         │   plus  lam x chamfer(tokens),  lam = 3.0
                                         ▼
                               re-ranked shortlist
                                         │   keep neighbours within 60 km of the new top-1
                                         ▼
                  similarity-weighted spherical mean,  temp = 0.1
                                         │
                                         ▼
                              pred_lat ,  pred_lng
```

Only the encoder holds weights. Everything downstream of it is arithmetic over stored vectors,
which is what lets a 5M-parameter network address a continent.


**THE ENCODER**
```
image 448x448 (input)
  └─ convnextv2_femto.fcmae_ft_in1k       ImageNet-pretrained init      4,848,240 params
       └─ feature map  384 x 14 x 14  =  384 x 196 tokens
  └─ GeM pooling, learned exponent p                                            1 param
  └─ Linear(384 -> 192)                                                    73,920 params
  └─ L2-normalise                        =>  192-d unit descriptor
                                                                total   4,922,161 params

  NOTE: the same Linear(384 -> 192) also projects the 196 spatial tokens, so chamfer
  re-ranking reuses the encoder and adds no weights of its own
```

The network learns a 192-dimensional descriptor where two images sit close together when they were
photographed close together. At inference the 11,758 labelled training images become a **gallery**
of descriptors, each stored next to its known coordinate. We embed a new photo, find its nearest
gallery neighbours, and decode its position from where those neighbours are.

Four decode stages sit between the raw nearest neighbour and the final coordinate. **None of them
adds a single parameter.** We measured each one on the held-out split. PCA-whitening is applied
to every row, including the first, so its own contribution is not a step in this ladder:

| stage | median | mean | < 25 km | < 200 km | < 750 km |
|---|---:|---:|---:|---:|---:|
| top-1 neighbour (multi-scale TTA, whitened) | 60.52 | 410.4 | 42.0% | 62.7% | 80.7% |
| + local-mode decode | 56.61 | 410.9 | 42.3% | 62.7% | 80.8% |
| + CSLS hubness correction | 49.21 | 403.4 | 43.3% | 63.5% | 81.3% |
| **+ chamfer re-ranking** | **34.16** | **359.5** | **47.4%** | **65.9%** | **83.5%** |

### 02. The parameter budget

| what | params | counted? |
|---|---:|---|
| `convnextv2_femto` backbone (all of it; the pretrained weights are trained, not frozen) | 4,848,240 | **yes** |
| GeM exponent `p` | 1 | **yes** |
| projection `Linear(384 → 192)` | 73,920 | **yes** |
| **submitted model** | **4,922,161** | **yes, and it is under 5,000,000 by 77,839** |

The whitening transform is a statistic of the gallery, not a learned weight. Gradient descent never
touches it, and we recompute it from the gallery at inference instead of shipping it.

---

## 03. Why retrieval, not regression

A network that regresses latitude and longitude has to keep the map inside its own weights. We didn't take it on faith either, we measured it: nine architecture
variants of coordinate regression and geo-cell classification, built along the standard PlaNet/ISN
lines, and every one of them produced not so much good result. Because under 5M parameters the weights just can't hold where every European street is.

So I stopped asking the network **where the photo was taken**. I ask it **which other photo this
one looks like**, then read the coordinate off that photo.

That moves the map out of the weights and into the gallery, because map is not a learnable parameters. So now a small network is genuinely good at, a sharp visual descriptor that
can tell two streets apart. 

I took this to the organisers before building anything on it, because two things needed settling:
whether a retrieval inference path was acceptable at all, and whether the gallery counted against
the parameter budget. Both came back confirmed.

---

## 04. The decode chain

Everything between the encoder and the final coordinate is below, in the order the diagram runs it.
None of these stages adds a parameter. Each one is here because I watched a particular failure
happen and built the stage to answer it.

- **Multi-scale TTA** (448/512/576, averaged and re-normalised). We train the network at 448.
  Evaluating higher gives GeM pooling a finer spatial grid.
- **PCA-whitening**, fitted on the gallery only, eigenvalue exponent 0.35 instead of a full 0.5.
  Without it a few dominant directions drown out everything else. Those directions are broad scene
  statistics, vegetation and light, and they swamp the fine cues that separate one place from the
  next.
- **CSLS** [17]. Some gallery images are *hubs*: they sit close to everything and come back for
  unrelated queries, so we suppress them by subtracting `beta = 0.25` times each gallery point's
  mean similarity to its own 10 nearest gallery neighbours, a quantity measured gallery-to-gallery
  and therefore **inductive**, using nothing whatsoever about the queries. We tested the
  transductive variant too, but it's worse at the equivalent stage.
- **Chamfer re-ranking**, the largest single gain at −15.0 km. A global descriptor pools the whole
  feature map into one vector, so two images can sit close in it while agreeing on nothing in
  particular. Comparing the *spatial token sets* asks a stricter question. Does some region of this
  image match some region of that one? That's the question that separates "another road in the same
  country" from "the same street". For the top-64 shortlist:

  ```
  cham(q,g) = 0.5 · ( mean_i max_j {q_i, g_j}  +  mean_j max_i {q_i, g_j} )
  score     = global_similarity  +  3.0 · cham(q,g)
  ```

  It's the local-matching idea of Patch-NetVLAD / DELG [19,20] cut down to its cheapest form. The
  tokens pass through **the same** `Linear(384 → 192)` as the global descriptor, so the stage
  introduces no new weights.
- **Local-mode decode.** The final coordinate is a similarity-weighted spherical mean of the
  re-ranked neighbours within 60 km of the new top-1. Restricting to that neighbourhood makes it a
  local **mode** and not a global mean. Otherwise one confident neighbour in the right place gets
  cancelled by a spread of wrong ones elsewhere in the country. We average on unit vectors, not
  degrees, so it stays correct across the antimeridian and at high latitude. That matters here:
  the dataset contains Iceland and northern Norway.

Whitening and the hubness term are fitted on the gallery alone. **Nothing about the test set enters
the pipeline**, and every query is decoded independently of every other query.

---

## 05. Environment and data

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
export GEO_DATA=/path/to/geo_dataset     # must contain train/, holdout_public/, train_labels.csv
```

Tested with **Python 3.12.3, CUDA 12.6, PyTorch 2.13.0+cu126, timm 1.0.29** on a single
**NVIDIA RTX 4000 Ada (20 GB)**. `requirements.txt` pins the exact versions we used.

**You need a CUDA GPU.** 8 GB is enough at the default `--bs 48`. The decoder in `geo.py` falls
back to CPU if it can't see a GPU, but training and embedding won't.

Expected data layout:

```
geo_dataset/
├── train/                11,758 labelled .jpg images
├── holdout_public/        2,400 unlabelled .jpg images
└── train_labels.csv       filename,country,iso,lat,lng   (11,758 rows + header)
```

**Network access.** `train.py` calls `timm.create_model(..., pretrained=True)` **once**, which
downloads the ImageNet weights for `convnextv2_femto.fcmae_ft_in1k` (about 20 MB) from the Hugging
Face hub, and that single call is the only network access anything in this pipeline makes. Set
`HF_HOME` to a warmed cache to run offline. `embed.py`, `predict.py` and `eval.py` all build the network with `pretrained=False`, so
**inference is fully offline**.

## 06. Files

| file | what it is |
|---|---|
| `geo.py` | network, geometry and retrieval decoder. The shared core, imported by everything else. |
| `train.py` | training, on the full dataset (`--val-frac 0`) or an 80/20 split (`--val-frac 0.2`) |
| `embed.py` | descriptor extraction: multi-scale TTA globals, plus per-token descriptors |
| `predict.py` | gallery + holdout → `predictions.csv` |
| `eval.py` | held-out scoring, the ablation ladder and the per-country breakdown |
| `model/full_data/final.pt` | the submitted weights, 19 MB, 4,922,161 parameters |
| `model/full_data/config.json` | exact arguments, parameter count and loss curve of that run |
| `predictions.csv` | 2,400 rows for `holdout_public/`, `filename,pred_lat,pred_lng` |
| `requirements.txt` | pinned dependencies, versions as tested |
| `writeup/Writeup.pdf` | the 1–2 page report, shipped alongside this directory |


## 07. Training

**Loss.** Take a batch and its pairwise great-circle distances `d_ij`. Build a target distribution
from the geography, then match the embedding's similarity distribution to it:

```
Q = softmax_j( -d_ij / τ )              τ = 15           where the images actually are(in kms)
P = softmax_j( (e_i · e_j) / temp )     temp   = 0.07    what the embedding thinks
loss = KL(Q ‖ P)                                         self-similarity masked out of both
```

This is a **soft-target contrastive loss**. Instead of splitting positives from negatives at some
arbitrary radius, every image in the batch counts as a positive in proportion to how close it
really is. So the gradient carries *how much* closer one neighbour is than another. That's the
signal a metric for geolocation needs.

**`τ = 15` is the single most important hyper-parameter in the project.** An earlier version
used 250 km. At 250 km the target can't distinguish 1 km from 40 km, so every nearby image looks
equally positive and the embedding learns only *which region* a photo came from. Never place
identity. Dropping to 15 km is what made retrieval work at all. It's the difference between the
126 km plateau and everything reported here.

**Batch sampling.** Only about a quarter of the images have any partner within 500 m. Draw a batch
at random and it usually contains no positive pair, and the loss degenerates into noise. So we build
each batch as **12 anchors × 4 images drawn from within 25 km of that anchor**. That guarantees true
positives, and it also fills the batch with the hard negatives that turn out to matter just as much,
meaning other places in the same region that look similar to the anchor but sit kilometres away from
it.

**Augmentation.** RandAugment (`rand-m7-n2-mstd0.5`) [13] plus random resized crop, scale
(0.5, 1.0). **Horizontal flip is disabled.** Flipping mirrors driving side, sign text and lane
direction, which are among the strongest country clues in Europe.

**Schedule.** AdamW (weight decay 0.01), OneCycle, 40 epochs, backbone at one third of the head
learning rate (`3e-4` head, `1e-4` backbone), AMP fp16. Steps per epoch scale with dataset size, so
the schedule completes exactly one cycle whether it trains on 80% or 100% of the data.

**Epoch selection: there is none.** We keep the last epoch, with the budget fixed at 40 in advance.
Saving the best epoch by validation score would leak the validation split into model selection and
make the reported number optimistic.

**Decode hyper-parameters**, frozen at these values for both `predict.py` and `eval.py`:

| flag | value | |
|---|---:|---|
| `--csls-k` | 10 | gallery neighbours defining the hubness term |
| `--csls-beta` | 0.25 | weight of the hubness subtraction |
| `--topk` | 64 | shortlist size handed to re-ranking |
| `--lam` | 3.0 | weight of the chamfer term |
| `--radius` | 60 km | neighbourhood around the top-1 used for decoding |
| `--temp` | 0.1 | softmax temperature on the similarity weights |


## 08. Producing `predictions.csv`

```bash
# 1. train on all 11,758 labelled images                                    ~67 min
python train.py --out model/full_data --epochs 40

# 2. descriptors for the gallery and for the holdout               ~4 min + ~1 min
python embed.py --ckpt model/full_data/final.pt --split train   --out gallery.npz
python embed.py --ckpt model/full_data/final.pt --split holdout --out holdout.npz

# 3. predictions.csv                                                          ~2 min
python predict.py --gallery gallery.npz --query holdout.npz --out predictions.csv
```

These exact commands produced the shipped `predictions.csv` from the shipped
`model/full_data/final.pt`. Steps 2 and 3 are deterministic given the checkpoint and will regenerate
it exactly. Step 1 won't. It involves fp16 and cuDNN kernel selection, so retraining lands close to
the shipped weights but not bit-identical with them.



## 09. Steps to Reproduce the project

The submitted model trains on all 11,758 images, so it has no held-out data of its own. What we
validate is the **recipe**, on a stratified 80/20 split of the labelled set:

```bash
python train.py --out model/val --val-frac 0.2 --epochs 40   # ~60 min
python embed.py --ckpt model/val/final.pt --split train --out val_emb.npz
python eval.py  --emb val_emb.npz
```

`eval.py` rebuilds the identical split from the same seed. It uses the 9,406 training images as
gallery and the 2,352 held-out images as queries, then prints the ablation ladder plus a per-country
breakdown. Three things make that number meaningful and not decorative:

- **`--seed` must match between `train.py` and `eval.py`.** Both default to 0. `geo.split_train_val`
  regenerates the split from the seed, stratified by country. We never store it, so it can't drift
  out of sync with the model.
- **`embed.py --split train` embeds all 11,758 images**, but `eval.py` uses only the 9,406 training
  rows as gallery. The held-out 20% is never a retrieval candidate, and whitening and hubness
  statistics are fitted on the gallery alone.
- **Nothing is tuned on the held-out split.** Every decode hyper-parameter is frozen at the value
  `predict.py` uses. `eval.py` contains no search. The epoch is fixed at 40 in advance (It's a hyprparameter though).

**Retraining reproduces this to within run-to-run variation, not exactly; fp16, cuDNN kernel selection
and sampler interaction all move it a little. The split itself is exactly reproducible. The
seed-based split and the stored split from the reported run agree on all 2,352 held-out images.

## 10. Verifying the submission

Two checks a grader can run without reading any of the code.

**Parameter count.** This is independent of anything `train.py` prints, and it runs against the
shipped weights:

```bash
python -c "
import torch, geo
net = geo.GeoNet(pretrained=False)
net.load_state_dict(torch.load('model/full_data/final.pt', map_location='cpu'))
n = sum(p.numel() for p in net.parameters())
print(f'{n:,} parameters   compliant={n <= 5_000_000}')
for name, m in [('backbone', net.bb), ('projection', net.fc)]:
    print(f'  {name:12s} {sum(p.numel() for p in m.parameters()):>10,}')
print(f'  {\"gem exponent\":12s} {net.p.numel():>10,}')
"
# 4,922,161 parameters   compliant=True
```

It counts the checkpoint that **actually ships**, so a smaller model than the one that produced
`predictions.csv` can't satisfy it.

**`predictions.csv` format**, against the exact specification in the challenge:

```bash
python -c "
import os, pandas as pd
d = pd.read_csv('predictions.csv')
exp = sorted(f for f in os.listdir(os.environ['GEO_DATA'] + '/holdout_public')
             if f.lower().endswith('.jpg'))
assert list(d.columns) == ['filename', 'pred_lat', 'pred_lng'], d.columns
assert len(d) == 2400, len(d)
assert sorted(d.filename) == exp, 'filenames do not match holdout_public/'
assert d.filename.is_unique and d.notna().all().all()
assert d.pred_lat.between(-90, 90).all() and d.pred_lng.between(-180, 180).all()
print(f'OK  {len(d)} rows  lat [{d.pred_lat.min():.3f}, {d.pred_lat.max():.3f}]'
      f'  lng [{d.pred_lng.min():.3f}, {d.pred_lng.max():.3f}]')
"
```

We write gallery embeddings **in `train_labels.csv` row order**, and both `predict.py` and `eval.py`
assert that order before using them. 


## 11. References

1. S. Mehta, M. Rastegari. *MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision
   Transformer.* ICLR, 2022.
2. T. Weyand, I. Kostrikov, J. Philbin. *PlaNet — Photo Geolocation with Convolutional Neural
   Networks.* ECCV, 2016. — the geo-cell classification baseline we tried and abandoned.
3. E. Müller-Budack, K. Pustu-Iren, R. Ewerth. *Geolocation Estimation of Photos using a
   Hierarchical Model and Scene Classification.* ECCV, 2018. — the hierarchical variant of the same.
4. P. H. Seo, T. Weyand, J. Sim, B. Han. *CPlaNet: Enhancing Image Geolocalization by
   Combinatorial Partitioning of Maps.* ECCV, 2018.
5. L. Haas, M. Skreta, S. Alberti, C. Finn. *PIGEON: Predicting Image Geolocations.* CVPR, 2024.
6. M. Tan, Q. V. Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*
   ICML, 2019.
7. B. Clark, P. Kerrigan, P. P. Kulkarni, V. V. Cepeda, M. Shah. *Where We Are and What We're
   Looking At: Query Based Worldwide Image Geo-localization Using Hierarchies and Scenes.*
   CVPR, 2023. — the attention-pooling variant we tried and dropped.
8. J. Hays, A. A. Efros. *IM2GPS: Estimating Geographic Information from a Single Image.*
   CVPR, 2008.
9. S. Woo et al. *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders.*
   CVPR, 2023. — the backbone and its FCMAE pre-training.
10. F. Radenović, G. Tolias, O. Chum. *Fine-tuning CNN Image Retrieval with No Human
    Annotation.* TPAMI, 2019. — GeM pooling.
11. A. van den Oord, Y. Li, O. Vinyals. *Representation Learning with Contrastive Predictive
    Coding.* arXiv:1807.03748, 2018. — the InfoNCE form our embedding similarity follows.
12. P. Khosla et al. *Supervised Contrastive Learning.* NeurIPS, 2020. — the fixed-radius
    positive/negative split our soft-target loss replaces.
13. E. D. Cubuk, B. Zoph, J. Shlens, Q. V. Le. *RandAugment: Practical Automated Data
    Augmentation with a Reduced Search Space.* NeurIPS, 2020.
14. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR, 2019. — AdamW.
15. L. N. Smith, N. Topin. *Super-convergence: Very Fast Training of Neural Networks using Large
    Learning Rates.* SPIE, 2019. — OneCycle.
16. H. Jégou, O. Chum. *Negative Evidences and Co-occurences in Image Retrieval: The Benefit of
    PCA and Whitening.* ECCV, 2012.
17. A. Conneau, G. Lample, M. Ranzato, L. Denoyer, H. Jégou. *Word Translation Without Parallel
    Data.* ICLR, 2018. — CSLS, introduced for bilingual lexicon induction and applied here to
    geographic retrieval.
18. M. Radovanović, A. Nanopoulos, M. Ivanović. *Hubs in Space: Popular Nearest Neighbors in
    High-dimensional Data.* JMLR, 11:2487–2531, 2010.
19. S. Hausler, S. Garg, M. Xu, M. Milford, T. Fischer. *Patch-NetVLAD: Multi-Scale Fusion of
    Locally-Global Descriptors for Place Recognition.* CVPR, 2021. — local-descriptor re-ranking of
    a global shortlist.
20. B. Cao, A. Araujo, J. Sim. *Unifying Deep Local and Global Features for Image Search.*
    ECCV, 2020. — DELG; the same retrieve-then-rerank structure.
21. O. Chum, J. Philbin, J. Sivic, M. Isard, A. Zisserman. *Total Recall: Automatic Query Expansion
    with a Generative Feature Model for Object Retrieval.* ICCV, 2007. — α-query expansion we
    tried and dropped.
22. Z. Zhong, L. Zheng, D. Cao, S. Li. *Re-ranking Person Re-identification with k-reciprocal
    Encoding.* CVPR, 2017. — k-reciprocal re-ranking we tried and dropped.
23. M. Wortsman, G. Ilharco, S. Y. Gadre, et al. *Model Soups: Averaging Weights of Multiple
    Fine-tuned Models Improves Accuracy without Increasing Inference Time.* ICML, 2022. — the
    "model soup" result we discarded as contaminated.

*Structure of this document:* G. A. A. Prana, C. Treude, F. Thung, T. Atapattu, D. Lo.
*Categorizing the Content of GitHub README Files.* Empirical Software Engineering 24, 2019,
1296–1327.

---

## 12. What did not work

We record these because the negative results are what shaped the method, and because most of the
time that went into this project went here.

- **Coordinate regression and geo-cell classification.** Nine architecture variants, including
  hierarchical country→cell→offset heads, multi-crop fusion, and label smoothing over neighbouring
  cells. Every one plateaued at 122–155 km regardless of backbone or resolution. An oracle analysis,
  bounding the error if the correct cell were always chosen, showed cell accuracy was the ceiling.
  Not resolution. That's what motivated abandoning regression entirely.
- **`τ = 250` in the retrieval loss.** Same architecture, same training, one hyper-parameter
  different, and it learns region identity instead of place identity.
- **Horizontal-flip augmentation.** Disabled throughout, because flipping mirrors driving side,
  sign text and lane direction, among the few cues that reliably separate European countries. This
  one is an assumption, not a result: we never ran the ablation.
- **k-reciprocal re-ranking, α-query expansion, label propagation, transductive whitening.** All
  neutral or worse. They redistribute retrieval mass among the neighbours already retrieved, which
  does nothing when the failure mode is that the right place is absent from the shortlist. CSLS
  helps because it corrects a different defect: hub gallery images crowding out correct ones before
  the shortlist is even formed.
- **Confidence-gated snapping** to the top-1 when the shortlist agrees. Exactly 0.00 km gain, which
  in hindsight is obvious: wherever the shortlist agrees strongly enough to trigger a snap, the
  local-mode decode was already returning essentially the top-1 anyway.
- **Higher input resolution** (512, 576 at training time). 30–60% more time per epoch for no
  reliable median improvement. Resolution was never the bottleneck. Shortlist recall was.
- **Weight-averaging the cross-validation checkpoints**, the "model soup". It appeared to give
  26 km. We dropped it as invalid instead of reporting it: each fold model is the only one blind to
  its own fold, so any average of two or more has seen every image in the validation set. The
  apparent gain was contamination.



