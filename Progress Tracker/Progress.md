# Progress Tracker

**How to use this file:** checklist of what's actually done vs. pending, organized by phase.
Detailed experiment numbers live in `Experiment_Design.md` (results log) — this file links to
experiment IDs (e.g. `T1.1`) rather than repeating scores, to avoid two files going out of sync.
Update this at the end of every work session.

**Current phase:** Planning / pre-implementation. No code or environment set up yet.
**Last updated:** 2026-08-17

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
- [ ] Are pretrained weights (e.g. ImageNet-pretrained backbone under 5M params) allowed, or
      is it train-from-scratch only? Currently assuming from-scratch — this affects Tier 2+
      choices significantly if the answer is "pretrained is fine."
- [ ] Does mixup/cutmix-style image blending count as "synthetic/generated data"? Currently
      treating it as disallowed.
- [ ] **Action:** send this email — not yet sent.

## Phase 3 — Environment & repo setup
- [ ] Set up a proper git repo scoped to `Final Project/` (currently `git status` resolves to
      the home directory as repo root, which is wrong — needs its own `git init` here plus a
      `.gitignore` excluding `geo_dataset/`). **Deferred by choice — will set up later.**
- [x] Set up Python environment: `.venv` created (Python 3.12), installed torch 2.13,
      torchvision 0.28, pandas, numpy, scikit-learn, matplotlib, PIL, scipy.
- [x] Pin versions into `requirements.txt` — generated via `pip freeze`.
- [x] Confirmed local compute: this machine is a **MacBook (Apple M5)**, not the RTX 4000 —
      MPS backend confirmed available (`torch.backends.mps.is_available() == True`).
      **Dual-machine workflow going forward**: local Mac/MPS for fast iteration and quick
      1-2 fold screening; RTX 4000 (CUDA, 16GB VRAM) for full 5-fold runs and final training.
- [ ] Confirm GPU access / environment parity on the RTX 4000 machine when we get there.

## Phase 4 — Protocol implementation (Experiment_Design.md Part 1)
- [ ] Geographic leakage check: pairwise-distance clustering analysis on `train_labels.csv`
      to check for near-duplicate locations across what would be train/val splits.
- [ ] Decide final CV strategy based on leakage check result: plain stratified 5-fold vs.
      group-stratified 5-fold.
- [ ] Generate and save fixed fold indices (`folds.json`) — reused by every experiment from
      here on so comparisons in the results log are apples-to-apples.
- [ ] Implement spherical-vector target conversion utilities (`lat,lng -> unit vector` and
      back) + angular/cosine loss function.
- [ ] Implement and run EDA baselines B0 (global mean) and B1 (oracle-country-centroid) —
      results go in `Experiment_Design.md` results log. B1 in particular tells us how much
      error is "wrong country" vs. "imprecise within country."
- [ ] Build reusable error-analysis tooling: predicted-vs-true map plot, country-confusion
      matrix, per-image error histogram, per-country median-km breakdown.
- [ ] Build automated param-compliance checker (instantiate model at real input res, assert
      `sum(p.numel())` ≤ 5,000,000) — run before training any candidate, not after.

## Phase 5 — Architecture experiments (Experiment_Design.md Part 2)
Two-stage screening per architecture: quick 1-2 fold pass to filter, full 5-fold only for the
shortlist. Status below tracks screening stage reached, not scores (see results log for numbers).

- [ ] T1.1 — naive CNN, direct spherical regression (pipeline proof-of-concept)
- [ ] T2.1 — MobileNetV3-Small (from scratch)
- [ ] T2.2 — ShuffleNetV2 (from scratch)
- [ ] T2.3 — custom narrow ResNet
- [ ] T3.1 — grid-cell classification + offset regression (best Tier 2 backbone)
- [ ] T3.2 — multi-task with country auxiliary head
- [ ] T4.1 — mixture density output
- [ ] T4.2 — tiny CNN+attention hybrid (MobileViT-XXS scale)
- [ ] T5.1 — two-model ensemble (combined ≤5M params, verified)
- [ ] T5.2 — test-time augmentation on best model

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
