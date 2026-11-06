# Progress Tracker

**How to use this file:** checklist of what's actually done vs. pending, organized by phase.
Detailed experiment numbers live in `Experiment_Design.md` (results log) — this file links to
experiment IDs (e.g. `T1.1`) rather than repeating scores, to avoid two files going out of sync.
Update this at the end of every work session.

**Current phase:** Phase 5 — Architecture experiments. **Best verified result: 149.5 ±
4.6km, full 5-fold CV** (T4.1c, shared global/local EfficientNet-B0 RA4 retrieval encoder +
supervised geographic contrastive continuation). Beats the previous best, T3.4's
182.9 ± 7.0km, by 33.4km. **Important correction (2026-08-20): the 118km leaderboard number
is a hard threshold for leaderboard-grading credit, not a soft reference** — recomputed the
honest, single-fixed-decoder (not per-fold-cherry-picked) CV number: **151.06 ± 3.74km**,
~33km still to close. Params **4,722,346** (compliant, 277,654 below the limit). T4.1c is a
two-stage recipe: stage 1 (T4.1b, 60 epochs, `backbone_lr=1e-4`) trains the shared 5-view (1
global + 4 local corner crops) retrieval encoder from an ImageNet-pretrained init; stage 2
(T4.1c, 20 epochs, `backbone_lr=2e-5`, 5x lower) continues with a supervised geo-contrastive
loss switched on (geo-balanced batches, weight 0.2, zero new params). Full 5-fold run
(2026-08-20, jobs 4067429-4067443): per-fold 153.45/151.79/144.44/144.56/153.28km, cell acc
43.7%, country acc 77.0% (`outputs/experiments/T4.1c_geo_supcon_5fold/cv_summary.json`).
Architecture diagram: `writeup/figures/architecture_t4_1c.tex` (compiled with `tectonic`).
**T4.1d (coarse country->region head, 3-level hierarchy) is architecture-complete and CPU-
verified but GPU-untested** — paused mid-session by a cluster GPU quota block (`AssocGrpGRES`,
account-wide, ~5h53m of GPU usage today across 13 jobs; confirmed NOT partition-specific by
testing a40/a100/rtxpro6k, all eventually blocked). Queue fully cleared, all sbatch scripts
now default to `a40` (was `rtxpro6k`) per explicit instruction. **Next session: resubmit the
T4.1d smoke test first** (see T4.1d checklist entry below for the exact command) before
anything else — do not assume the quota has reset without checking `squeue` for
`AssocGrpGRES` first.
**Last updated:** 2026-08-20

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
- [x] **T3.3 + T3.6 combined — REGRESSED, 2026-08-19, reverted.** Combined 210.4±7.8km vs.
      T3.1's 193.0±4.2km. Isolated: T3.6-only (230.3km) was the primary culprit, worse than
      the combined run and the only condition that also hurt country acc; T3.3-only (200.5km)
      was closer to neutral. Code reverted to plain T3.1; archived (not deleted) in
      `failed_experiments/T3.3_T3.6_soft_labels_attention_pool/` — see that folder's README
      and `Experiment_Design.md` Tier 3 for full numbers/reasoning. **Project convention from
      here on: archive failed experiment code+artifacts in `failed_experiments/`, don't
      delete.**
- [x] **T3.4 — combinatorial/multi-partition cells — SUCCESS, 2026-08-19.** New best:
      **182.9 ± 7.0km** (full 5-fold), beats T3.1 on every fold (see `Experiment_Design.md`
      for full numbers/reasoning). `src/protocol/geo_cells_multi.py`,
      `src/models/multi_partition_geo_classifier.py`, `src/train_multi_partition.py`.
- [ ] T3.5 — three-level country->region->cell hierarchy (see `literature.md` §1, ISNs)
- [~] T3.4 K=3 partitions — code ready, cells built, param-checked (4,394,410, compliant).
      Smoke test queued then cancelled (GPU cluster fully saturated, user paused work).
      Not yet run. Next: resubmit smoke test, then fold-0 screen vs. K=2's 182.4km.
- [x] **T4.1 — soft-hierarchical geographic retrieval refinement — DONE, fold-0 screen
      166.85km / 160.42km with mode decoder.** New files: `src/models/retrieval_geo_classifier.py`,
      `src/geo/retrieval.py`, `src/train_retrieval_geo.py`, and
      `slurm/train_retrieval_geo_fold.sbatch`. Architecture: EfficientNet-B0 RA4 backbone;
      K=3 raw cell heads; country and offset heads; 256-D normalized embedding; geography-
      distribution loss based on pairwise haversine distance; full soft `P(country) ×
      P(cell|country)` scoring of fold-training gallery locations; embedding-similarity
      refinement; score-weighted spherical top-k output. Parameter count 4,722,346.
      Crucial preprocessing correction: previous EfficientNet runs reused MobileViT's
      unnormalized `[0,1]` transform even though `efficientnet_b0.ra_in1k` expects ImageNet
      mean/std. T4.1 uses the selected checkpoint's exact `timm` preprocessing; RA4 expects
      mean/std `(0.5,0.5,0.5)`. Smoke job 4061496 passed; 60-epoch fold-0 screen job
      4061688 completed at 166.85km, improved to 160.42km with a zero-parameter mode decoder.
- [x] **T4.1b — shared global/local multi-view retrieval encoder — DONE, fold-0 screen
      155.07km.** `src/models/multiview_retrieval_geo_classifier.py`,
      `src/train_multiview_retrieval_geo.py`, `slurm/train_multiview_retrieval_geo_fold.sbatch`.
      Same backbone/heads as T4.1, but the backbone is shared across 5 views per image (1
      global 288px + 4 local 64%-width corner crops), fused without new params. Fold-0: job
      4063141, 155.07km at epoch 51/60, country acc 77.38%, cell acc 43.51%.
- [x] **T4.1c — supervised geographic contrastive continuation — DONE, FULL 5-FOLD CV:
      149.5 ± 4.6km, NEW BEST RESULT (2026-08-20).** Continues each fold's T4.1b checkpoint
      for 20 epochs at 5x lower LR with a supervised geo-contrastive loss on (geo-balanced
      batches, weight 0.2, zero new params) — see `Experiment_Design.md` Tier 4 for the full
      per-fold table and architecture diagram (`writeup/figures/architecture_t4_1c.tex`).
      Beats T3.4 (182.9 ± 7.0km) by 33.4km; gap to the ~114-120km reference target down to
      ~29.5-35.5km. Params unchanged at 4,722,346.
- [~] **T4.1d — coarse country->region head (3-level hierarchy) — BUILT, GPU-untested (paused
      on cluster GPU quota, 2026-08-20).** Motivation: cell classification accuracy has been
      stuck at ~42-44% across T3.1/T3.4/T4.1/T4.1c despite every surrounding mechanism
      improving — `literature.md`'s top-ranked remaining candidate (ISNs/Clark et al.
      multi-granularity hierarchy). New files (per this project's "never edit a working
      baseline in place" convention): `src/protocol/geo_cells_region.py` (coarse per-country
      partition, MIN=150/MAX=400, gives 2 regions/country = 24 total, vs. the fine
      partitions' 96 cells/country), `src/models/region_retrieval_geo_classifier.py`
      (T4.1c's encoder + one new `region_head`, 30,744 params), `src/train_region_retrieval_geo.py`
      (continues from a trained T4.1c checkpoint via `strict=False` — region_head initializes
      fresh, everything else keeps its trained weights), `slurm/train_region_retrieval_geo_fold.sbatch`.
      Region head does NOT feed the offset target/blended centroid (a region is too coarse a
      per-image locator) — it only adds a `region_score_weight * log P(region|country)` term
      to retrieval/hierarchy scoring, plus its own masked CE loss. **Verified before any GPU
      time**: syntax/imports, real fold-0 data loads correctly (region_id in [0,23], exactly
      2/country), checkpoint continuation loads with `strict=False` producing exactly the
      expected missing keys (`region_head.weight/bias`, nothing else), forward pass shapes
      all correct, param count **4,753,090** (compliant, 246,910 headroom). **GPU smoke test
      (job 4068334, then resubmitted as 4068348/4068350 while chasing a cluster GPU quota
      issue — see below) never actually completed** — paused before any epoch ran. No
      trained result exists yet; this is architecture-complete, GPU-blocked, not evidence of
      anything working or not working. **Next session: resubmit the 2-epoch smoke test
      first** (`FOLD=0 EXP_NAME=T4.1d_region_smoke_v2 EPOCHS=2 INIT_CHECKPOINT=outputs/experiments/T4.1c_geo_supcon_5fold/fold0/best_model.pt
      sbatch slurm/train_region_retrieval_geo_fold.sbatch`), confirm region loss decreases
      and region_acc is reasonable (near 50% or above, since 2-way per-country is close to
      chance at 50%), then run the real ~20-epoch fold-0 screen before committing to a full
      5-fold run.
- [ ] T4.2 — mixture density output (optional stretch; lower priority than retrieval)
- [ ] T5.1 — two-model ensemble (combined ≤5M params, verified) (optional stretch)
- [ ] T5.2 — test-time augmentation on best model (optional stretch)

### Researched backbone shortlist for the final hierarchical model

These are **candidates, not measured project results**. Totals include the real country,
offset and K-partition heads and were instantiated locally rather than copied from papers.

| Priority | Exact `timm` backbone | K=2 total | K=3 total | Status / rationale |
|---:|---|---:|---:|---|
| 1 | `efficientnet_b0.ra4_e3600_r224_in1k` | 4,271,434 | 4,394,410 | Selected for T4.1; same proven B0 architecture, stronger checkpoint |
| 2 | `repvit_m0_9.dist_450e_in1k` | 4,797,102 | 4,834,062 | Strongest genuinely different near-budget challenger |
| 3 | `efficientformerv2_s0.snap_dist_in1k` | 3,282,718 | 3,299,710 | Local/global hybrid with ample head budget |
| 4 | `convnextv2_atto.fcmae_ft_in1k` | 3,453,526 | 3,484,342 | Modern convolutional features for texture, roads and architecture |
| 5 | `mobilenetv4_conv_small.e2400_r224_in1k` | 2,690,990 | 2,783,246 | Efficient modern CNN; leaves room for richer decoding |
| 6 | `efficientnet_es.ra_in1k` | 4,421,278 | 4,544,254 | Low-risk higher-capacity EfficientNet alternative |
| 7 | `edgenext_x_small.in1k` | 2,183,562 | 2,202,090 | Native 256px local/global model for fine street cues |
| 8 | `mobilevitv2_100.cvnets_in1k` | 4,494,519 | 4,543,767 | Stronger MobileViT generation; native 256px |
| 9 | `mobilenetv3_large_100.ra_in1k` | 4,399,998 | 4,492,254 | Fits unlike the unnecessarily weak Small candidate |
| 10 | `fastvit_t8.apple_dist_in1k` at 256px | 3,415,646 | 3,489,470 | Native-resolution retest only; already 213.4km at 224px |

Explicit exclusions: TinyViT-5M exceeds the complete-model limit (5,168,706 at K=3),
EfficientNet-B1 and larger exceed budget, and MobileViT-XS/FastViT-T8 at 224px are already
measured below EfficientNet-B0. Backbone swaps follow T4.1 retrieval validation because a
backbone alone is unlikely to close the full 68.9km gap.

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
- **2026-08-19 (cont.)**: Fold-0 screen came in at 225.3km; switched default resolution to
  224px for a controlled 3-backbone comparison (EfficientNet-B0, FastViT-T8, MobileViT-XS),
  all full 5-fold. **EfficientNet-B0 won: 193.0 ± 4.2km**, new best result, beats B1 by
  95.1km. User asked to push further and requested a literature review before picking the
  next architecture direction, rather than guessing. Researched the actual research lineage
  for CNN-based photo geolocation (PlaNet, CPlaNet, ISNs, TransLocator, PIGEON, GeoCLIP,
  MDN-based approaches) via web search for accurate citations, wrote `literature.md` —
  filtered every idea through our actual constraints (11.7k images, 5M params, no external
  data) rather than assuming anything ports as-is. Cross-referenced findings against our own
  diagnosis (cell accuracy 41.8% is the bottleneck, not offset precision) and ranked
  candidates by fit + cost: **T3.3 (PIGEON's haversine-smoothed soft cell labels)** is the
  clear next step — zero new params, no architecture change, directly targets cell-boundary
  misclassification, which plausibly explains a real chunk of the current 58.2% wrong-cell
  rate. T3.4 (CPlaNet's combinatorial multi-partition cells) queued as the follow-up if T3.3
  doesn't close enough of the gap alone; T3.5 (ISNs' three-level hierarchy) as a higher-cost
  fallback. Scene-branch/segmentation ideas (ISNs, TransLocator) considered and explicitly
  not recommended — need labels/tools we don't have without touching the external-data
  boundary. Updated `Experiment_Design.md` Tier 3 (T3.1 marked done with its real result,
  T3.3-T3.5 added) and this file's checklist accordingly. Next: implement and screen T3.3.
- **2026-08-19 (cont.)**: User asked what architecture change specifically (not just T3.3's
  loss change) would push results further. Fetched Clark et al. 2023's actual architecture
  (not just search snippets) via WebFetch to get it right before implementing: Swin backbone
  (not portable), learned-query cross-attention over spatial tokens instead of GAP (T3.6, the
  portable core idea), shared-then-specialized decoder staging, 7-level S2 hierarchy (+1.5%
  @1km going 3->7 levels, supports T3.5 eventually), scene queries (not portable, external
  data). Logged full technical detail into `literature.md` for accuracy. Implemented both
  T3.3 and T3.6: new `src/models/attention_pool.py` (`QueryAttentionPool` -- 2 queries,
  1 shared cross-attn layer + 2 specialized FFN branches, token dim reduced to 128 to stay
  in budget -- naive full-width attention would have cost ~6.5M params by itself, over budget
  alone), `hierarchical_geo_classifier.py` gained a `pooling={gap,attention}` switch (default
  attention), `train_hierarchical.py` gained `soft_cell_targets()` (haversine-smoothed target
  distribution, `tau=50km`, a tuned guess not from the paper) and `--soft-cell-labels`
  (default on). Verified param compliance for all 3 backbones (EfficientNet-B0: 4,450,794,
  still compliant) and ran a full CPU sanity pass (forward/backward, soft-target sums to 1,
  zero mass outside the true country's cells) before touching GPU.
  User asked to run all 5 folds in parallel (not the usual fold-0-then-decide gate) for
  speed, accepting the tradeoff explicitly. Smoke test hit real infra friction first: `a40`
  and `a100` partitions were both fully saturated (cluster-wide, confirmed via `sinfo`/
  `scontrol`), so tried `rtxpro6k` (had genuinely free GPUs per `scontrol show node`) --
  that job ran 10 minutes with ZERO output (not even the shell's own first `echo`) before
  TIMEOUT killed it. Root-caused via the job's own `nvidia-smi -L` output: `rtxpro6k` nodes
  have an **NVIDIA RTX PRO 6000 Blackwell** GPU, a much newer architecture than every
  previously-validated node (`a40`/`a100`, Ada/Ampere) -- our pinned `torch 2.13.0+cu130`
  almost certainly has no precompiled kernels for it, so the first CUDA call likely fell into
  slow JIT compilation, looking exactly like a hang. Reverted to `a40` (queued ~15min, then
  ran) rather than gamble on an unvalidated GPU generation -- smoke test PASSED cleanly there
  (job 4049825, 76.6s/epoch at batch=16, correct param count, full train/eval loop).
  **Full 5-fold result (jobs 4049847-4049851): 210.4 ± 7.8km, cell acc 36.1%, country acc
  72.8% -- a REGRESSION vs. T3.1's 193.0 ± 4.2km / cell acc 41.8%.** Consistent across every
  fold (cell acc landed in a tight 35.0-35.5% band vs. T3.1's 41-43% band, near-zero
  variance), country accuracy exactly flat -- damage is real and specifically localized to
  cell classification, not noise. This is the exact attribution problem flagged before
  agreeing to run both changes together: can't tell from this result alone whether T3.3,
  T3.6, or their interaction is responsible. Submitted two isolation screens, fold-0 only:
  T3.6-only (job 4053329, hard labels + attention pooling) and T3.3-only (job 4053330, soft
  labels + plain GAP), both to compare against T3.1's actual fold-0 number (191.7km, cell
  42.6%). Next: read those two results, diagnose which change (or interaction) caused the
  regression, decide whether either is salvageable (e.g. a smaller/better-initialized
  attention pool, a different tau) or whether to drop both and move to T3.4/T3.5 instead.
- **2026-08-19 (cont.)**: Both isolation screens converged. **T3.6-only (attention pool, hard
  labels): 230.3km** (epoch 86, cell acc 37.5%, country acc 70.9%) — worse than even the
  combined run (218.1km), and the only condition of the four where country accuracy also
  dropped, clearly implicating T3.6 as the primary cause rather than an interaction effect.
  **T3.3-only (soft labels, plain GAP): 200.5km** (epoch 84, cell acc 37.7%, country acc
  72.9%) — much closer to neutral (+8.8km vs. baseline), not proven but not clearly harmful
  either. Full 4-way table (baseline / T3.3-only / combined / T3.6-only) logged in
  `Experiment_Design.md` Tier 3. User decided to revert both to the T3.1 baseline and,
  **starting now, archive failed experiment code+artifacts instead of deleting them** (a
  change from the earlier StarNet-S1 rollback, which was deleted outright) — created
  `failed_experiments/` at repo root as the standing convention for this and every future
  failed attempt. Reverted `src/models/hierarchical_geo_classifier.py`,
  `src/train_hierarchical.py`, `src/protocol/param_check.py` to their exact pre-T3.3/T3.6
  content (not just defaulted the new flags off) so the active pipeline has no trace of the
  regressed code path. Moved `src/models/attention_pool.py` and all four T3.3/T3.6 sbatch
  scripts into `failed_experiments/T3.3_T3.6_soft_labels_attention_pool/`, plus the four
  experiment output directories (smoke test, combined 5-fold, both isolation screens), plus a
  README summarizing what was tried, the full numbers, the diagnosis, and what to try if
  revisited (T3.3 with a tuned tau, not T3.6 as designed). Verified the revert is clean:
  `python -m py_compile` passes on all three reverted files, and
  `check_compliance_hierarchical_model` for EfficientNet-B0 now reports exactly
  4,148,458 params again (the original T3.1 count, confirming no attention-pool params
  survived). Next: T3.4 (combinatorial multi-partition cells) — see `Experiment_Design.md`.
- **2026-08-19 (cont.)**: Implemented T3.4. Threaded an optional `rng` param through
  `geo_cells.py`'s existing `build_node`/`build_fold` (randomized split-axis choice; `None`
  default reproduces the canonical T3.1 partition exactly, verified) rather than duplicating
  the partition-building logic. New `geo_cells_multi.py` builds K=2 independent per-country
  partitions per fold (same MIN=40/MAX=120 as T3.1); verified real structural diversity
  (e.g. Germany splits on `lng` in partition 0 but `lat` in partition 1) before trusting it.
  Extended `dataset.py` additively (`load_fold_cells_multi`, `blended_centroid_np` --
  spherical mean, consistent with the project's existing convention -- and a
  `multi_cell_info` path on `GeoDataset` alongside, not replacing, the existing `cell_info`
  path). New model `MultiPartitionGeoClassifierRegressor`
  (`src/models/multi_partition_geo_classifier.py`) in its own file rather than a flag on
  `hierarchical_geo_classifier.py` -- explicit decision to not repeat the T3.3/T3.6 pattern
  of modifying the working baseline in place. New `check_compliance_multi_partition_model`
  (param_check.py, additive) and `train_multi_partition.py` (new script, parallel structure
  to train_hierarchical.py). Verified compliant (4,271,434 params) and ran a full CPU sanity
  pass before GPU -- notably, the untrained model's random-init numbers landed almost exactly
  at the theoretical random baseline (cell acc ~1% = 1/96, country acc ~8% = 1/12), a good
  sign the wiring was correct before spending any GPU time. Smoke test (job 4054609) passed
  cleanly on `a40`. Fold-0 screen (job 4054693) beat T3.1's fold-0 baseline (191.7km) partway
  through training and held the lead through convergence: **180.96km at epoch 75** -- a real
  win, not a mid-training artifact. Ran the full 5-fold grid (jobs 4055791-4055795, one job
  per fold, all 5 running concurrently on separate `a40` GPUs -- user explicitly confirmed
  this is the intended workflow, not everything bundled into one job). **Result: 182.9 ±
  7.0km, beating T3.1's 193.0 ± 4.2km on every single fold** (fold deltas: -9.3, -4.8, -14.7,
  -10.7, -10.9km) -- confirms this is a robust, general improvement, not one lucky fold.
  Cell acc (42.1%) and country acc (72.7%) both landed essentially unchanged from T3.1
  (41.8%/72.8%), confirming the entire gain comes from the blending mechanism itself (two
  independently-imperfect coarse cell picks triangulating a better point), exactly the
  CPlaNet mechanism this was designed to test -- not from the model getting better at the
  underlying classification task. **New best result, gap to the ~114km reference target down
  to 68.9km** (from T3.1's 79km). Added `slurm/train_multi_partition_fold.sbatch` as the
  reusable template (the actual 5-fold submission used an equivalent `sbatch --wrap` for
  speed; the template is on record for reproducibility). Next: decide between pushing K to 3
  partitions (this run used the minimum K=2 to test the mechanism cheaply first) or moving to
  T3.5 -- open decision, not yet made.
- **2026-08-20 — T4.1/T4.1b retrieval refinement:** T4.1 fold-0 completed at **166.85km**
  (job 4061688, epoch 52/60, `retrieval_w4_k10`). A zero-parameter geographic mode decoder
  evaluated on that frozen checkpoint improved it again to **160.42km** (`w=2`, top-50,
  250km hard neighborhood; job 4062906). Implemented T4.1b as a separate reproducible path:
  one shared EfficientNet-B0 RA4 processes a global view plus four 64%-width corner crops;
  global/local features are fused without new weights, each batch contains four examples per
  geographic cell, a consistency loss aligns the five descriptors, and retrieval mixes
  scene similarity with the strongest local-detail match before mode decoding. Exact count
  remains **4,722,346 params** (277,654 under the 5M limit); all views/labels/gallery entries
  use only provided data. CPU shape/loss/compliance checks passed. Two-epoch GPU smoke job
  4063014 passed end-to-end (21.1GiB peak, 59-64s/epoch); its deliberately undertrained score
  improved 431.6 -> 305.8km and is only a pipeline check. Full 60-epoch fold-0 screen
  submitted as **job 4063141**, experiment `T4.1b_multiview_mode_fold0_screen`. **Completed
  in 1h01m: 155.07km at epoch 51/60**, 465.37km mean, 54.46% <200km, 81.51% <750km,
  country accuracy 77.38%, and mean cell accuracy 43.51%. Best decoder was
  `mode_w2_l0.25_r250`. This improves T4.1 mode decoding by 5.35km and T3.4 fold-0 by
  25.89km, but remains 35.07km above the 120km target. Next refinement must strengthen the
  geographic retrieval embedding directly; further decoder-only tuning is unlikely to close
  that remaining gap.
- **2026-08-20 — T4.1c direct geographic metric refinement:** Added a supervised
  contrastive objective to the T4.1b embedding. Geo-balanced batches provide four samples
  per partition-0 cell; positive strength is proportional to how many of the three
  independent cells two images share, while other sampled cells act as negatives. This adds
  zero parameters. Continued the T4.1b best checkpoint for 20 epochs at 5x lower learning
  rates (job 4065137, 21m49s). **Result: 150.79km at continuation epoch 14**, 450.67km mean,
  54.51% <200km, 81.97% <750km, hard median 167.04km, country accuracy 77.76%, cell accuracy
  43.51%; best decoder `mode_w4_l0_r150`. This improves T4.1b by 4.28km and T4.1 by 9.64km
  with the same **4,722,346** parameters. It is a genuine improvement but still 30.79km
  above the 120km target, so no full 5-fold claim is made.
- **2026-08-20 (cont.) — 118km clarified as a hard leaderboard-grading threshold, not a soft
  reference; recalibrated priorities accordingly.** User corrected the framing: leaderboard
  grading credit requires reaching ~118km, so architecture work continues rather than moving
  to Phase 6 packaging. Before trusting the 149.5km full-CV number as the real target gap,
  checked whether it was inflated by per-fold decoder selection (each fold's `fold_result.json`
  picked its own best-scoring decoder config using that fold's own validation labels — not
  available on the real holdout). Recomputed the honest number from each fold's already-saved
  `decoder_stats` using ONE fixed config (`plain_w4_l0.25_k10`) across all 5 folds: **151.06 ±
  3.74km** — only 1.6km worse than the cherry-picked number, so the earlier result wasn't
  meaningfully optimistic, but the honest number is now the one to report/close the gap on.
  Remaining gap to 118km: ~33km.
- **2026-08-20 (cont.) — built and launched the missing Phase 4 error-analysis tooling.**
  Wrote `src/error_analysis_t4_1c.py` (per-image/per-country error breakdown + country
  confusion matrix, all 5 folds out-of-fold, fixed decoder config) to find WHERE the
  remaining ~33km is concentrated before guessing at the next architecture move. Submitted as
  `slurm/error_analysis_t4_1c.sbatch` (job 4068290) — this job never completed this session
  (see GPU quota entry below); rerun next session before drawing conclusions from it.
- **2026-08-20 (cont.) — recommended and built T4.1d, a coarse country->region head (3-level
  hierarchy).** User asked directly what architectural change would close the gap
  "brutally," not incrementally. Recommendation, grounded in the project's own numbers: cell
  classification accuracy has been stuck at 41.8%->42.1%->43.7% across T3.1/T3.4/T4.1c (three
  completely different architectures) while every score improvement came from better
  *combining* imperfect cell predictions, never from the classifier itself improving —
  `literature.md`'s own top-ranked untried candidate (ISNs/Clark et al.'s multi-granularity
  hierarchy) targets exactly this. Built it as described in the T4.1d checklist entry above
  (`geo_cells_region.py`, `region_retrieval_geo_classifier.py`, `train_region_retrieval_geo.py`,
  new sbatch script) — fully CPU-verified (real fold-0 data, checkpoint-continuation loading,
  forward shapes, param compliance at 4,753,090) but never reached a GPU smoke test this
  session due to the quota issue below.
- **2026-08-20 (cont.) — hit and diagnosed a cluster-wide GPU quota block; paused GPU work for
  the day.** After submitting the T4.1d smoke test and the error-analysis job, both sat
  `PENDING (AssocGrpGRES)` on `rtxpro6k`. Initially misdiagnosed as partition-specific
  (a40/a100 accepted the jobs immediately with normal `Priority`/`None` reasons) and moved
  jobs between partitions repeatedly to route around it -- but every partition eventually
  flagged the same jobs `AssocGrpGRES` within ~30-60s of submission, including with ZERO jobs
  actually running (confirmed via `sacct`/`squeue` -- not a concurrent-usage limit). Checked
  cumulative usage: ~5h53m of GPU time across 13 completed jobs today (4x ~1h01m T4.1b
  stage-1 runs, 5x ~21m T4.1c stage-2 runs, plus smaller smoke tests) -- concluded this is a
  genuine account-level GPU-hour/GPU-count budget exhausted for the day, not a bug in any
  job or a partition quirk. No admin-level visibility available (`sacctmgr`/`sshare` queries
  returned empty limit fields for this association) to confirm the exact reset policy. User
  decided to pause GPU work for today rather than keep retrying blind. Cancelled all pending
  jobs, queue fully cleared. **Per explicit instruction, `rtxpro6k` is no longer used at
  all** -- updated `slurm/train_multiview_retrieval_geo_fold.sbatch`,
  `slurm/train_region_retrieval_geo_fold.sbatch`, and `slurm/error_analysis_t4_1c.sbatch` to
  default to `--partition=a40 --gres=gpu:a40:1` (was `rtxpro6k`). **Next session: check
  `squeue` for `AssocGrpGRES` before assuming the quota has reset** -- don't resubmit blind.
- **2026-08-20 (cont.) — T4.1c full 5-fold CV, run in parallel via SLURM job chaining:**
  User asked to run T4.1c as full 5-fold CV (all folds in parallel) and to produce a
  compiled architecture diagram for it. Since T4.1c is a two-stage recipe (T4.1b base
  training -> T4.1c contrastive continuation) and only fold 0 had a T4.1b checkpoint from
  the prior session, submitted T4.1b stage-1 for folds 1-4 (`jobs 4067429-4067432`, default
  hyperparams, EPOCHS=60) and copied fold 0's existing checkpoint into the same unified
  `T4.1b_multiview_mode_5fold` experiment dir (no wasted GPU time re-training it). Submitted
  T4.1c stage-2 for all 5 folds immediately, chaining folds 1-4 via SLURM
  `--dependency=afterok:<stage1_job_id>` so each fold's contrastive continuation auto-starts
  the instant its own stage-1 job completes -- no polling/babysitting needed, and true
  fold-level parallelism given cluster GPU availability (`jobs 4067439-4067443`,
  `T4.1c_geo_supcon_5fold`). All 9 jobs ran cleanly on `rtxpro6k`, no failures. Stage-1
  (folds 1-4) completed in ~1h01m each; stage-2 (all 5 folds) completed in ~20-22min each.
  **Aggregated via `src/aggregate_fold_results.py`: 149.5 ± 4.6km full 5-fold CV median**
  (per-fold: 153.45 / 151.79 / 144.44 / 144.56 / 153.28km), mean 449.5km, cell acc 43.7%,
  country acc 77.0% (`outputs/experiments/T4.1c_geo_supcon_5fold/cv_summary.json`). **New
  best result overall**, beating T3.4's 182.9 ± 7.0km by 33.4km -- tight cross-fold std
  (4.6km) confirms this is a real, general improvement. Gap to the ~114-120km reference
  target narrowed from 68.9km (T3.4) to ~29.5-35.5km. Honest caveat for the writeup: folds 1
  and 4 peaked at continuation epoch 1 and 3 (of 20), so most of the gain comes from turning
  the contrastive loss on at all, not from training it for long -- worth investigating
  whether a shorter continuation schedule or different LR would be equally effective at
  lower cost, if revisited. Also built `writeup/figures/architecture_t4_1c.tex` (new
  standalone TikZ figure following the existing `architecture_t3_4.tex` convention),
  covering the shared 5-view encoder, all six training losses (highlighting the new
  supervised geo-contrastive term), and the full inference-time retrieval/mode-decode path;
  compiled cleanly with `~/.local/bin/tectonic` after fixing two real issues (missing
  `amsmath` for `\tfrac`, and a `label=`-on-background-layer bug in `fit` nodes that clipped
  long section-header labels at the page edge -- fixed by promoting labels to plain
  foreground nodes anchored off the fit box's `.north`, a pattern worth reusing in any future
  figure with long group labels). Next: decide whether to push further (tune contrastive
  `tau`/weight/schedule given the early-peaking folds, or the still-unrun T3.4 K=3
  partitions) or treat 149.5km as the final architecture and move to Phase 6 packaging.
