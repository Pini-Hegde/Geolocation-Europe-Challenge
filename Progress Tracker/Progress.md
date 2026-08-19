# Progress Tracker

**How to use this file:** checklist of what's actually done vs. pending, organized by phase.
Detailed experiment numbers live in `Experiment_Design.md` (results log) — this file links to
experiment IDs (e.g. `T1.1`) rather than repeating scores, to avoid two files going out of sync.
Update this at the end of every work session.

**Current phase:** Phase 5 — Architecture experiments. Hierarchical country->cell classifier +
offset regression, 3-backbone full 5-fold comparison at 224px done: **EfficientNet-B0 wins,
193.0 ± 4.2km** (best result so far, beats B1's 288.1km oracle by 95.1km). FastViT-T8
213.4±10.1km, MobileViT-XS 240.7±12.1km. Deciding next step (finalize EfficientNet-B0, or
push further toward the ~114km reference target).
**Last updated:** 2026-08-19

---

## Phase 0 — Understand the challenge
- [x] Read `CHALLENGE.md`, confirmed task: predict `(lat, lng)` for street-level photos from
      12 European countries, scored by median haversine km on a private holdout.
- [x] Confirmed hard constraints: ≤5,000,000 params in submitted model; train only on
      provided data (no external/scraped/synthetic data).
- [x] Explored dataset on disk: 11,758 labelled images in `train/`, 2,400 unlabelled in
      `holdout_public/`, `train_labels.csv` with `filename, country, iso, lat, lng`.
      Country counts ~1,000 each except Iceland (960), Poland (934), Belarus (864).
- [x] Confirmed exact deliverables: code + trained weights + `predictions.csv` + 
      `requirements.txt` + `README.md`, plus a 1-2 page LaTeX writeup, zipped together.
- [x] Confirmed grading split: 50% method/code quality, 35% writeup reasoning, 15% leaderboard.

## Phase 1 — Planning docs
- [x] Created `Progress Tracker/` with `Progress.md`, `Experiment_Design.md`, `Project_plan.md`.
- [x] Designed and finalized `Experiment_Design.md`:
  - [x] Methodology/Protocol section (leakage check, k-fold CV, two-stage screening,
        spherical-vector target parameterization, EDA baselines, param-compliance check,
        augmentation policy, error-analysis tooling).
  - [x] Architecture tiers T1 (naive) through T5 (ensembling), all designed to respect the
        5M-param / from-scratch constraints.
- [ ] Design and finalize `Project_plan.md` (end-to-end step-by-step flow of the project) — **next up**.

## Phase 2 — Open questions to resolve with organizers (blocking some decisions)
- [x] Are pretrained weights (e.g. ImageNet-pretrained backbone under 5M params) allowed, or
      is it train-from-scratch only? **Resolved (2026-08-17): pretrained weights are allowed,
      any source, as long as the submitted model is ≤5,000,000 params. Fine-tuning the
      pretrained model on the provided `train/` data is allowed.** This opens up Tier 2+
      considerably — see `Experiment_Design.md` for the updated tier list.
- [ ] Does mixup/cutmix-style image blending count as "synthetic/generated data"? Email sent
      to organizers (2026-08-18), awaiting response. Still treating it as disallowed until
      confirmed.

## Phase 3 — Environment & repo setup
- [ ] Set up a proper git repo scoped to `Final Project/` (currently `git status` resolves to
      the home directory as repo root, which is wrong — needs its own `git init` here plus a
      `.gitignore` excluding `geo_dataset/`). **Deferred by choice — will set up later.**
- [x] Set up Python environment: `.venv` created (Python 3.12), installed torch 2.13,
      torchvision 0.28, pandas, numpy, scikit-learn, matplotlib, PIL, scipy.
- [x] Pin versions into `requirements.txt` — generated via `pip freeze`.
- [x] Confirmed local compute: earlier sessions ran on a **MacBook (Apple M5)** — MPS backend
      confirmed available (`torch.backends.mps.is_available() == True`).
      **Dual-machine workflow going forward**: local Mac/MPS for fast iteration and quick
      1-2 fold screening; RTX 4000 (CUDA, 16GB VRAM) for full 5-fold runs and final training.
- [x] Confirmed GPU access on the RTX 4000 machine (2026-08-18): this session is running
      directly on it — `torch.cuda.is_available() == True`, `torch.cuda.get_device_name(0)`
      == "NVIDIA RTX 4000 Ada Generation". **Caveat:** `requirements.txt` was regenerated via
      `pip freeze` on *this* (CUDA) machine for the `timm` addition, so it now pins
      CUDA-specific wheels (`nvidia-*`, `cuda-*`, `triton`) that won't install on the Mac's
      MPS environment. Fine for now (mid-experimentation), but flagged as something to fix
      before final packaging (Phase 6) — e.g. regenerate on/for whichever machine trains the
      final model, or hand-prune the CUDA-only lines.
- [x] **Third compute environment added (2026-08-18): NHR@FAU Alex HPC cluster**
      (`alex.nhr.fau.de`, `$CLUSTER=alex`). Discovered mid-session that a Claude Code session
      here was actually running on Alex's login node (`alex2`), not the RTX 4000 — the two are
      unrelated machines that happen to share this project's home-directory path. Alex login
      nodes have **no GPU at all** (no `nvidia-smi`); real training must go through SLURM
      (`sinfo` shows `a40`/`a100`/`rtxpro6k`/`a100mig` GPU partitions, `a40` used as default —
      largest pool, 44 nodes). `.venv` did not exist on Alex (whatever ran there before must
      have been on the RTX 4000 or another machine); rebuilt fresh via
      `module load python/3.12-base && python3 -m venv .venv` + `pip install -r
      requirements.txt` — installed cleanly (`torch 2.13.0+cu130`, `timm 1.0.28`), confirming
      the CUDA-13 pins are compatible with Alex's A40 nodes. **Found and fixed a real gotcha:
      Alex's GPU compute nodes have no outbound internet** — a first SLURM job hung 5 minutes
      and TIMEOUT'd trying to download pretrained weights from Hugging Face Hub mid-job. Fixed
      by pre-downloading weights into the local HF cache from the (internet-connected) login
      node, then setting `export HF_HUB_OFFLINE=1` in every sbatch script. Added
      `slurm/smoke_test.sbatch` (5min sanity check: GPU visible, param-compliance, dataloader,
      forward pass — run before any real job to catch env issues in minutes, not hours) and
      `slurm/train_t2_1_5fold.sbatch` (the real training job template — 1x GPU, `HF_HUB_OFFLINE=1`,
      `--mail-type=END,FAIL`). Confirmed empirically (via `who`/`w` on the login node — several
      other accounts had 2-5 concurrent `pts` sessions) that Alex allows unlimited concurrent
      SSH sessions per user, and SLURM jobs run fully detached from the submitting shell/SSH
      session once `sbatch`'d — verified in practice: job 4041460 (see Phase 5) ran to
      completion under `slurmd` on its compute node regardless of this session's own state,
      unlike the earlier login-node `nohup`/`setsid` attempts which had no such protection.
      **Open flag, not yet resolved:** `outputs/experiments/T2.1_mobilevit_xs_5fold_attempt1_died_ep30/`
      is a failed non-SLURM 5-fold attempt (died ~epoch 30, killed by the login node) that is
      already git-committed from an earlier session (part of commit `c6af406`) — left in place
      pending a decision on whether to keep it as a documented failure for the writeup or
      remove it.

## Phase 4 — Protocol implementation (Experiment_Design.md Part 1)
- [x] Geographic leakage check: pairwise-distance clustering analysis on `train_labels.csv`
      to check for near-duplicate locations across what would be train/val splits.
      `src/protocol/leakage_check.py` — 24.56% of points in a multi-point cluster at 500m
      (worst: Iceland 47.5%, Norway 45.4%). See `Experiment_Design.md` §1 for full result.
- [x] Decide final CV strategy based on leakage check result: plain stratified 5-fold vs.
      group-stratified 5-fold. **Decided: group-stratified 5-fold** (group = 500m cluster,
      stratify by country) — leakage was real, not noise.
- [x] Generate and save fixed fold indices (`folds.json`) — reused by every experiment from
      here on so comparisons in the results log are apples-to-apples.
      `src/protocol/make_folds.py` — group-stratified 5-fold (`StratifiedGroupKFold`, group =
      500m cluster from the leakage check, stratify by country, seed=42). Group integrity
      verified (no cluster split across folds); stratification tight (max per-country/fold
      share deviation 0.001 from ideal). Output: `outputs/protocol/folds/folds.json`
      (filename -> fold id + generation metadata), `fold_country_counts.csv`, `summary.md`.
- [x] Implement spherical-vector target conversion utilities (`lat,lng -> unit vector` and
      back) + angular/cosine loss function. `src/geo/spherical.py` (numpy + torch
      implementations: `latlon_to_unit_vector_*`, `unit_vector_to_latlon_*`, `angular_loss`,
      `haversine_km_*` for reporting). Verified in `src/protocol/target_check.py`: round-trip
      on all 11,758 real points (max error 2.8e-12 km), pole edge cases, the ±180° wraparound
      fix demonstrated concretely (true dist 222.4km vs. naive raw-degree squared error
      128,164), loss bounds (identical=0, orthogonal=1, antipodal=2), and torch/numpy
      agreement — all checks passed. `outputs/protocol/target_parameterization/`.
- [x] Implement and run EDA baselines B0 (global mean) and B1 (oracle-country-centroid) —
      results in `Experiment_Design.md` results log. `src/protocol/eda_baselines.py`, scored
      on the fixed group-stratified 5-fold split, centroids computed as spherical means
      (reuses `src/geo/spherical.py`, not naive degree averages). **B0 median 1217.8 ± 9.1km,
      B1 median 288.1 ± 4.2km** — country-level signal is worth most of the error budget
      (1218→288km), but real within-country precision work remains even with country solved
      (B1 <200km rate only 28%). Worst within-country error: Norway/Spain/Turkey (large,
      sparse); best: Iceland/Belarus/Poland (compact) — motivates Tier 3 grid-cell design.
      `outputs/protocol/eda_baselines/`. B2 (weighted random guess) deferred, not run.
- [ ] Build reusable error-analysis tooling: predicted-vs-true map plot, country-confusion
      matrix, per-image error histogram, per-country median-km breakdown.
- [x] Build automated param-compliance checker (instantiate model at real input res, assert
      `sum(p.numel())` ≤ 5,000,000) — run before training any candidate, not after.
      `src/protocol/param_check.py`, wired as a gate at the start of `src/train.py`.
      MobileViT-XS: 1,934,003 params (compliant). `outputs/protocol/param_check/`.

## Phase 5 — Architecture experiments (Experiment_Design.md Part 2)
Two-stage screening per architecture: quick 1-2 fold pass to filter, full 5-fold only for the
shortlist. Status below tracks screening stage reached, not scores (see results log for numbers).

**Note (2026-08-18):** Tier 1 (naive from-scratch CNN) dropped from the plan — decided its
only purpose (pipeline smoke test) is equally well served by building the real Tier 2 model
first, so it added a step without adding information. List below updated to match the
current `Experiment_Design.md` tier list (MobileViT-XS promoted to default primary
candidate, `timm` added as a dependency).

- [x] T2.1 — MobileViT-XS, ImageNet-pretrained (`timm`) — **new default primary candidate**.
      Full 5-fold CV done (2026-08-18, on Alex/`a40`, SLURM job 4041460): **459.9 ± 19.4km**
      median — confirms the fold-0 screen (441.8km) wasn't optimistic, tight spread across
      folds. Still **171.8km short of B1** (288.1km oracle ceiling) — see `Experiment_Design.md`
      results log for the full story (underfit → overfit → fixed → full CV) and per-fold
      breakdown. Rules out further Tier 2 tuning as the next step in isolation — T2.1's
      `<200km` rate (17.8%) is *worse* than B1's (28%), a baseline that doesn't look at the
      image at all, pointing at country/region-level confusion rather than a backbone-precision
      ceiling. **Next architecture direction is open — under active discussion (2026-08-19).**
- [ ] T2.1b — attention-pooling head ablation (learned query cross-attention vs. plain GAP).
- [ ] T2.2 — MobileNetV3-Small, ImageNet-pretrained (comparison point).
- [ ] T2.3 — GhostNet 1.0x, ImageNet-pretrained (comparison point).
- [ ] T2.4 — ShuffleNetV2, ImageNet-pretrained (comparison point).
- [ ] T2.5 — best of T2.1-2.4, from scratch (ablation: pretraining value).
- [ ] **T3.1 — hierarchical country -> within-country-cell classifier + offset regression.**
      (An earlier flat-classifier + soft-country-gate version was built, screened, and rolled
      back 2026-08-19 — see Session Log.) New design, in progress (2026-08-19):
      - **Backbone: MobileViT-XS**, deliberately kept the same as T2.1 (not swapped again) —
        isolates formulation + resolution as the tested variables, avoiding last round's
        "changed everything at once" attribution problem.
      - **Cells built per-country** (`src/protocol/geo_cells.py`, rewritten): each of the 12
        countries partitioned independently (same KD-tree median split, MIN=40/MAX=120), so
        every cell belongs to exactly one country by construction — no cross-country cell can
        exist. Result: 96 cells/fold, 8 cells/country, ~100 training images/cell.
      - **Hard hierarchical masking, not a soft bias** (`src/models/hierarchical_geo_classifier.py`):
        country head predicts country; cell logits are masked to only that country's cells
        (large negative value, not -inf, for AMP stability) before loss/argmax — true country
        used for masking during training (teacher forcing), predicted country used at eval/
        inference. This structurally prevents cross-country cell picks given a correct country
        prediction, unlike the rolled-back version's learned gate which only reduced their rate.
      - **384px input** (up from T2.1/the rolled-back attempt's 256px) — the one part of this
        design that's an educated guess rather than a measured fix; dataset ships native
        512x512 images, fine cues (signage text, plate style) may not survive 256px.
      - Params: **1,975,198** (compliant, ~2M under the 5M budget — heads are tiny next to the
        backbone). Verified at both 256px and 384px (param count is resolution-independent for
        this backbone, confirmed).
      - CPU sanity check + GPU smoke test (job 4043921, COMPLETED in ~1min) both passed.
      - Fold-0 screen (job 4043929, 384px, MobileViT-XS) **completed 2026-08-19: 225.3km best
        median** (epoch 85), country acc 72.7%, cell acc 37.3%, `<200km` 46.0%, `<750km`
        77.3% — beats the rolled-back attempt's best (247.1km) and its `<750km` (71-74%).
        Converged far faster too (68% country acc by epoch 7 vs ~90 epochs before).
      - **Backbone comparison launched 2026-08-19**: switched default input resolution to
        **224px** (was 384px) — verified via `timm.data.resolve_data_config` that this is
        EfficientNet-B0's actual native/pretrained resolution, close to MobileViT-XS's 256px
        recipe, but **not** FastViT's (that's 256px) — used uniformly anyway so the backbone
        comparison itself is controlled, with the FastViT mismatch flagged honestly rather
        than hidden. Added `efficientnet_b0.ra_in1k` (4,148,458 params total — notably closer
        to the 5M ceiling than the other candidates) and `fastvit_t8.apple_dist_in1k` (the
        smallest FastViT variant that fits budget — `fastvit_t12` alone is 6.5M, over budget)
        — both verified compliant at 224px. Rewrote the training script to run **one fold per
        SLURM job** (`slurm/train_hierarchical_fold.sbatch`, parameterized by
        `BACKBONE`/`FOLD`/`EXP_NAME` env vars) instead of one job looping over all folds, so
        all 15 (3 backbones x 5 folds) run in parallel instead of sequentially. Added
        `src/aggregate_fold_results.py` to compute the real cross-fold mean/std afterward
        from each job's independent `fold_result.json` (the in-script `cv_summary.json`
        write is now guarded to only fire for multi-fold single-job runs, so separate
        per-fold jobs sharing an `exp_name` can't silently overwrite each other's summary).
        Smoke-tested both new backbones on GPU (jobs 4044272, 4044273, both passed) before
        submitting the full grid (jobs 4044280-4044294). **MobileViT-XS full 5-fold result
        (2026-08-19): 240.7 ± 12.1km median**, mean 566.7km, cell acc 36.0%, country acc
        71.1% (`src/aggregate_fold_results.py`; tight std confirms the fold-0 screen's
        225.3km wasn't a lucky fold). Beats B1 (288.1km) by 47.4km.
        **EfficientNet-B0 full 5-fold result (2026-08-19): 193.0 ± 4.2km median**, mean
        532.4km, cell acc 41.8%, country acc 72.8% — **47.7km better than MobileViT-XS**,
        and a much tighter std (4.2km vs 12.1km, very consistent across folds). **New best
        full-CV reportable number.** The extra backbone capacity (4.15M vs 1.98M params)
        appears to directly help the diagnosed bottleneck (cell classification accuracy: 41.8%
        vs 36.0%). **FastViT-T8 full 5-fold result: 213.4 ± 10.1km median**, mean 552.0km,
        cell acc 39.1%, country acc 71.8% — beats MobileViT-XS (240.7km) despite training at
        a resolution mismatched to its own native pretraining (224px used here vs FastViT's
        actual 256px native/pretrained resolution, flagged earlier) — suggests FastViT is a
        genuinely competitive architecture that might close more ground at its native
        resolution, though not the current priority given EfficientNet-B0's clear lead.
        **Final 3-way backbone comparison (all hierarchical, 224px, full 5-fold):**
        EfficientNet-B0 193.0±4.2km (best) > FastViT-T8 213.4±10.1km > MobileViT-XS
        240.7±12.1km. All three beat B1 (288.1km). Gap to the ~114km reference target
        narrowed from 174.8km (B1) to 79km (EfficientNet-B0).
- [ ] T3.2 — multi-task with country auxiliary head
- [ ] T4.1 — mixture density output (optional stretch)
- [ ] T5.1 — two-model ensemble (combined ≤5M params, verified) (optional stretch)
- [ ] T5.2 — test-time augmentation on best model (optional stretch)

## Phase 6 — Final model & submission package
- [ ] Select final architecture based on full 5-fold results + writeup narrative value
      (grading weighs reasoning/method quality over raw leaderboard rank).
- [ ] Train final model on full `train/` set (not just CV folds), save weights.
- [ ] Generate `predictions.csv` for all 2,400 `holdout_public/` images, exact header
      `filename,pred_lat,pred_lng`, verify row count == 2,400 before packaging.
- [ ] Write `README.md`: exact setup/train/predict commands, final param count, training time.
- [ ] Finalize `requirements.txt`.
- [ ] Write 1-2 page LaTeX writeup: experiments tried (including failures), what worked, why —
      minimal intro/background, per grading rubric. Draw directly from
      `Experiment_Design.md` results log + error analysis findings.
- [ ] Package code + weights + `predictions.csv` + `requirements.txt` + `README.md` + writeup
      PDF into the final submission `.zip`.
- [ ] Final sanity pass: re-read `CHALLENGE.md` submission checklist end-to-end against the
      zip contents before submitting.

---

## Session Log
*(short dated entries — what was decided/done each session, for continuity across sessions)*

- **2026-08-17**: Explored dataset and `CHALLENGE.md`, set up `Progress Tracker/` folder.
  Discussed and finalized `Experiment_Design.md` — reframed from a plain architecture list
  into a methodology-first design (leakage check, k-fold CV, spherical target
  parameterization, EDA baselines) with architecture tiers as a secondary section, per
  agreement to think like a researcher rather than pick backbones off a leaderboard.
- **2026-08-17**: Resolved the pretrained-weights question directly with the project owner
  (no email needed): pretrained backbones of any origin are allowed, provided the submitted
  model is ≤5M params, and fine-tuning on the provided `train/` data is allowed. Updated
  `Experiment_Design.md` constraints and Tier 2+ accordingly. Mixup/synthetic-data question
  still open.
- **2026-08-17**: Ran the geographic leakage check (`src/protocol/leakage_check.py`, first
  code written in the project). Found real clustering — 24.56% of points within 500m of
  another point, worst in Iceland/Norway — so **CV strategy is locked to group-stratified
  5-fold** (plain stratified is ruled out). Cluster assignments saved to
  `outputs/protocol/leakage_check/clusters_500m.csv` for use as the group key.
- **2026-08-18**: Sent the mixup/cutmix synthetic-data question to organizers by email;
  awaiting response, still treating it as disallowed until confirmed. Generated fixed CV
  folds (`src/protocol/make_folds.py`): group-stratified 5-fold via `StratifiedGroupKFold`
  (group = 500m cluster, stratify by country, seed=42). Group integrity verified, stratification
  tight (0.001 max deviation). `outputs/protocol/folds/folds.json` is now the canonical split
  every experiment from here reads. Implemented and verified spherical-vector target
  utilities (`src/geo/spherical.py` + `src/protocol/target_check.py`) — round-trip,
  pole, ±180° wraparound, and loss-bound checks all pass on real data. Discussed
  whether to skip straight to a strong pretrained model instead of B0/B1 — decided to keep
  B0/B1 (near-zero cost, diagnostic value for the writeup) but skip the from-scratch Tier 1
  CNN and go straight to a Tier 2 pretrained backbone as the first real model. Ran B0/B1
  (`src/protocol/eda_baselines.py`): B0 median 1217.8±9.1km, B1 median 288.1±4.2km — logged
  in `Experiment_Design.md` results log. B2 deferred.
- **2026-08-18 (cont.)**: Discussed training a complex from-scratch CNN+attention model.
  Assessed and pushed back: attention lacks conv's locality/translation bias and is data-
  hungry, 11.7k images is small for training it from scratch, and it cuts against the
  project's own "complexity must earn its place" principle. Compared pretrained backbone
  options beyond MobileNetV3 (ShuffleNetV2, MnasNet, GhostNet, MobileViT-XS/XXS, EdgeNeXt,
  FastViT) and **decided on MobileViT-XS** (hybrid CNN+transformer, `timm`,
  `mobilevit_xs.cvnets_in1k`) as the new default primary Tier 2 candidate — gets the
  attention benefit in an already-pretrained, proven form rather than training it from
  scratch. Verified via `timm`: feature extractor ~1.93M params, 384-dim pooled output,
  256x256 input — well under the 5M budget. Installed `timm` (regenerated
  `requirements.txt` via `pip freeze` — now pins CUDA-specific packages, see Phase 3 note
  above). Updated `Experiment_Design.md` Tier 2 (T2.1 = MobileViT-XS + T2.1b attention-
  pooling-head ablation, old candidates renumbered as comparison points) and Tier 4.2
  (superseded, folded into T2.1). Also confirmed this session runs directly on the RTX 4000
  (CUDA) machine, not the Mac. Next: build the Tier 2.1 pipeline — dataset/dataloader,
  param-compliance check, MobileViT-XS + regression head, training loop with `angular_loss`,
  quick 1-2 fold screening run.
- **2026-08-18 (cont.)**: Built the full Tier 2.1 pipeline (`src/data/dataset.py`,
  `src/models/backbone_regressor.py`, `src/protocol/param_check.py`, `src/train.py`) and ran
  fold-0 quick-screen training on GPU (RTX 4000). Iterative debugging, each step diagnosed
  via evidence, not guessing:
  1. 15 epochs → 851km median, confirmed **underfit** (train-set eval showed ~858km too, no
     train/val gap) — too few gradient steps for the domain shift from ImageNet classification.
  2. 60 epochs → 611.6km, plateaued; diagnosed as an **LR-schedule artifact** (cosine decay
     already near-zero) rather than a capacity ceiling.
  3. 100 epochs + higher LR → 492.8km, but train-set eval revealed **overfitting** (train
     240.9km vs. val 492.8km, 251.9km gap).
  4. Added regularization (dropout=0.3 before head, weight_decay 1e-4→5e-4, wider
     augmentation) and re-ran 100 epochs → **441.8km median, gap shrank to 144.1km** (43%
     reduction) — confirms genuinely better generalization, not a lucky number. Still 153.7km
     short of B1 (288.1km).
  Also hit and fixed two infra issues along the way: (a) Python stdout block-buffers when
  redirected to a file, so a backgrounded run can look silent for a long time despite
  training correctly — worked around with `python -u`; (b) `nohup ... & disown` does *not*
  survive whatever's tearing down sessions in this environment (correlated with the user's
  SSH dropping twice this session) because `disown` doesn't create a new process
  session — switched to `setsid`, verified independent SID, and confirmed empirically it
  survived a subsequent SSH drop that killed the monitor watching it but not the training
  itself. Results logged in `Experiment_Design.md`. Next: full 5-fold CV with the
  regularized config for the real reportable T2.1 number.
- **2026-08-18 (cont.)**: Ran the full 5-fold CV, this time properly via SLURM instead of a
  backgrounded login-node process. Discovered this session's shell is actually on **NHR@FAU's
  Alex HPC cluster** (`alex2.nhr.fau.de` login node), not the RTX 4000 — the two share this
  project's directory path but are unrelated machines; Alex login nodes have no GPU at all, so
  a prior attempt at the full 5-fold run directly on the login node (untracked in this file,
  `outputs/experiments/T2.1_mobilevit_xs_5fold_attempt1_died_ep30/`, already git-committed)
  had died silently around epoch 30. Set up the Alex environment from scratch (`.venv` didn't
  exist there): `module load python/3.12-base`, fresh venv, `pip install -r requirements.txt`
  — installed cleanly on Alex's A40 nodes. Wrote `slurm/smoke_test.sbatch` and ran it first
  (caught a real bug: GPU compute nodes have no outbound internet, so `timm`'s pretrained-
  weight download hung and TIMEOUT'd after 5 min on the first attempt; fixed by pre-caching
  weights via the login node + `HF_HUB_OFFLINE=1`) before committing to the real ~2.5h job.
  Wrote `slurm/train_t2_1_5fold.sbatch` (1x `a40` GPU, 8h limit, same regularized config as
  the fold-0 screen, `--mail-type=END,FAIL`) and submitted it (job 4041460) — ran to
  `COMPLETED` in 2h32m with no issues, confirmed unaffected by session state throughout
  (verified this is a real property of `sbatch` jobs, not an assumption — they run under
  `slurmd` on the compute node independent of the submitting shell/SSH session, unlike the
  earlier `nohup`/`setsid` login-node attempts). **Result: 459.9 ± 19.4km full 5-fold CV
  median** — logged in `Experiment_Design.md`. Still short of B1 (288.1km) by 171.8km. Next:
  discuss with the user which direction to take before starting the next tier/subclass —
  T2.1b ablation, T2.2-2.4 comparison backbones, or a Tier 3 reformulation.
- **2026-08-18/19**: Explored a Tier 3 candidate — density-adaptive geo-cell classification +
  offset regression (KD-tree cell partitioning, StarNet-S1 backbone; a country-auxiliary-head
  variant was also built to address a cross-country-confusion failure mode found via error
  analysis). Screened on 2 folds; results were in the ~244-255km median range, beating B1
  (288.1km) but not by a large enough or clearly-architecture-motivated margin to commit to a
  full 5-fold run. **Rolled back 2026-08-19** — all code/artifacts/tracker detail for this
  attempt removed, per-decision to reconsider the architecture direction from scratch rather
  than continue iterating on this specific approach. State reverted to the T2.1 baseline
  above. Next: discuss alternative architecture candidates.
- **2026-08-19 (cont.)**: Discussed why the rolled-back attempt plateaued: error analysis had
  shown correct-cell error was already 66-84km (near the target the user's friend reportedly
  hit, ~114km) — the bottleneck was cell classification accuracy (~34-36%), not offset
  precision. Decided on a new design targeting that specific bottleneck: hierarchical
  (not soft-gated) country->cell classification, country-respecting cells (not a global
  partition), and testing 384px input (untested variable). Backbone deliberately kept at
  MobileViT-XS rather than swapped again, to isolate formulation+resolution as the two new
  variables. Confirmed param budget: full model (backbone + country head + cell head + offset
  head) is 1,975,198 params — ~2M of the 5M budget unused, heads are negligible next to the
  backbone. Built the full pipeline: rewrote `src/protocol/geo_cells.py` for per-country
  partitioning (96 cells/fold, 8/country, ~100 images/cell), added local-offset utilities
  back to `src/geo/spherical.py`, extended `src/data/dataset.py` with resolution flexibility
  + cell/country targets, new model `HierarchicalGeoClassifierRegressor` with hard masking
  (teacher-forced by true country in training, self-predicted country at eval), new
  `check_compliance_hierarchical_model`, new `src/train_hierarchical.py`. Verified locally on
  CPU (masking logic explicitly checked: masked-out logits are exactly -1e4), then via GPU
  smoke test (job 4043921, COMPLETED in ~1min). Submitted the fold-0 screen (job 4043929, 100
  epochs, 384px) — epoch 1 already at 48.2% country acc / 14.1% cell acc, well above the
  rolled-back version's equivalent-epoch numbers. Result pending.
