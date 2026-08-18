# Project Plan — End-to-End Flow

**Purpose of this file:** the *order of operations and decision logic* for the whole project,
top to bottom — what depends on what, where we deliberately stop to make a decision, and
where the plan is meant to loop back on itself rather than run once and finish. Task-level
checkboxes live in `Progress.md`; methodology detail and experiment results live in
`Experiment_Design.md`. This file is the map that connects them.

## Guiding principles (apply to every phase below)
1. **Protocol before architecture.** No experiment result counts until it's measured on the
   agreed CV protocol (`Experiment_Design.md` Part 1). Architecture and methodology are
   separate axes — don't let backbone excitement skip the measurement foundation.
2. **The architecture list is a living document, not a fixed plan.** Tiers 1-5 in
   `Experiment_Design.md` are our current best guess, not a contract — candidates get added,
   dropped, or reordered as error analysis reveals what the actual problem is. Phase D below
   is explicitly a loop for this reason.
3. **Every result is logged with reasoning, not just a number** — the results log doubles as
   raw material for the writeup, so "why did this work/fail" gets written down at experiment
   time, not reconstructed from memory at the end.
4. **Constraint checks (params ≤5M, from-scratch, no synthetic data) happen before training,
   not after** — verify, don't assume.
5. **Complexity must earn its place.** This is graded (50% method/code quality, 35% writeup
   reasoning, 15% leaderboard) and the rubric explicitly rewards a simple, well-argued idea
   over a tuned-but-unexplained complex one. Tier 1-3 in `Experiment_Design.md` are the core
   path; Tier 4-5 are optional stretch goals, added only when evidence (error analysis)
   specifically motivates them — not by default. Prefer clean, reproducible, explainable code
   over maximal sophistication.

---

## Pipeline overview

```
A. Problem & Data Understanding        [DONE]
        |
B. Environment & Repo Setup
        |
C. Protocol Foundation                  <-- gate: nothing in D starts until this is solid
   (leakage check -> CV strategy -> fixed folds ->
    target parameterization -> EDA baselines ->
    error-analysis tooling -> param-compliance checker)
        |
D. Architecture Experimentation LOOP  <----------------------+
   (pick candidate -> compliance check -> quick screen ->    |
    shortlist? -> full 5-fold -> log + error analysis  ------+  (loop: findings can
    -> decide next candidate)                                    add/drop/reprioritize
        |                                                        candidates, or send us
        | (loop exits when time/marginal-gain budget says stop)  back to C if a protocol
        v                                                        issue is discovered)
E. Model Selection & Finalization
        |
F. Submission Artifact Generation (predictions.csv, README, requirements.txt)
        |
G. Writeup (LaTeX)  -- drafted continuously alongside D, not written cold at the end
        |
H. Final Packaging & Submission Verification
```

---

## Phase A — Problem & Data Understanding — **DONE**
Read `CHALLENGE.md`, understood constraints and scoring, explored dataset structure and
class balance. See `Progress.md` Phase 0 for the checklist.

## Phase B — Environment & Repository Setup
**Objective:** a working, reproducible environment before any code is written, since
`requirements.txt` and `README.md` setup instructions are graded deliverables — get this
right once instead of reconstructing it at submission time.

- Project-scoped git repo with a `.gitignore` excluding `geo_dataset/` (currently the repo
  root resolves to the home directory, which is wrong — fix before writing any code).
- Python environment: torch/torchvision, PIL, pandas, numpy, scikit-learn (k-fold), matplotlib
  (error analysis).
- Confirm GPU access (RTX 4000, 16GB VRAM) for training runs.
- **Gate:** don't proceed to Phase C until `import torch; torch.cuda.is_available()` (or
  equivalent) is confirmed working end to end.

## Phase C — Protocol Foundation
**Objective:** build the measurement infrastructure and sanity baselines that every later
experiment depends on. This is the highest-leverage phase — a mistake here (e.g. undetected
leakage) invalidates every result built on top of it.

Order matters within this phase:
1. **Geographic leakage check first** — its outcome determines the CV strategy, so nothing
   downstream can be finalized before it runs.
2. **Decide & lock the CV strategy** (stratified vs. group-stratified 5-fold), generate and
   save fixed fold indices.
3. **Target parameterization utilities** (lat/lng ⇄ unit vector, angular loss) — used from the
   very first trained model onward, not introduced later.
4. **EDA baselines (B0, B1, B2)** — establishes the floor and tells us how much of the error
   budget is "country-level" vs. "within-country precision," which directly informs which
   Phase D tiers are worth prioritizing.
5. **Error-analysis tooling** (map plots, country-confusion matrix, error histograms,
   per-country breakdown) — built once here, reused for every experiment after.
6. **Automated param-compliance checker** — gate for every model trained from this point on.
- **Gate:** Phase D does not start until all six of the above are working and their outputs
  (fold indices file, baseline numbers, tooling scripts) exist on disk.

## Phase D — Architecture Experimentation (the loop)
**Objective:** find the best-performing model within budget, while producing an honest,
well-reasoned experiment trail for the writeup. This phase is explicitly cyclical:

```
pick a candidate (from Experiment_Design.md tiers, or a new idea from error analysis)
  -> run param-compliance check
  -> quick screen (1-2 folds)
  -> worth pursuing? --no--> log briefly, move to next candidate
       |
      yes
       v
  -> full 5-fold CV run
  -> log results + run error-analysis tooling
  -> does the error pattern suggest a new candidate, or that an existing
     tier idea should be dropped/reprioritized? --> update Experiment_Design.md
  -> repeat
```
- Candidates are **not locked to the current Tier 1-5 list** — if, say, Tier 2 error analysis
  shows most error is Poland/Belarus confusion, that's grounds to prioritize or invent a
  candidate that addresses that specifically, ahead of working through tiers in order.
- **Exit criterion for the loop:** diminishing returns (new candidates stop beating the
  current best by a meaningful margin relative to their added complexity/training cost) or
  the point where remaining time needs to shift to Phases E-H. This is a judgment call we
  make together, not a fixed number of experiments.
- All results accumulate in `Experiment_Design.md`'s results log; `Progress.md` Phase 5
  tracks which candidates have reached which screening stage.

## Phase E — Model Selection & Finalization
**Objective:** pick the model we actually submit.
- Compare shortlisted candidates on full 5-fold results (mean ± std, not just best fold).
- Weigh **narrative value for the writeup** alongside raw score — per the grading rubric
  (50% method/code quality, 35% writeup reasoning, 15% leaderboard), a well-justified model
  that's slightly behind the leaderboard leader can be the right choice.
- Re-verify param compliance on the exact final architecture/input resolution.
- Retrain the selected architecture on the **full** `train/` set (not just CV folds) for the
  submitted weights, since CV folds each withheld 20% of the data.

## Phase F — Submission Artifact Generation
- Generate `predictions.csv` for all 2,400 `holdout_public/` images — verify exact header and
  row count before moving on.
- Write `README.md` (setup/train/predict commands, final param count, training time) and
  finalize `requirements.txt` against the actual environment used.

## Phase G — Writeup (LaTeX)
**Objective:** should be largely a compilation exercise by this point, not a cold start —
draft sections incrementally as Phase D produces results, rather than writing everything
after Phase F. Content pulled directly from `Experiment_Design.md`'s results log and error
analysis findings: what was tried (including failures), what worked, why, with references
for any technique adapted from prior work (spherical target parameterization, grid-cell
classification, etc.).

## Phase H — Final Packaging & Submission Verification
- Assemble the `.zip`: code, trained weights, `predictions.csv`, `requirements.txt`,
  `README.md`, writeup PDF.
- Final pass: re-read `CHALLENGE.md`'s submission requirements line by line against the zip
  contents before submitting — cheap insurance against a missed formatting detail.

---

## Open decision points still ahead
- **Phase D direction, resolved 2026-08-18 (discussion only, not yet built):** after T2.1's full
  5-fold result (459.9±19.4km, still 171.8km short of B1's 288.1km oracle ceiling, and its
  `<200km` rate worse than B1's despite B1 ignoring the image entirely), decided to go straight
  to **T3.1 task reformulation** (adaptive density-based geo-cell classification + offset
  regression) rather than further Tier 2 backbone tuning — see `Experiment_Design.md` Tier 3 and
  `Progress.md` Phase 5 for full reasoning. **Backbone finalized as StarNet-S1** (`timm`, 2.9M
  params, 73.5% ImageNet top-1, CVPR 2024 "star operation" — a third architecture lineage
  distinct from both the depthwise-separable-conv family and MobileViT's attention hybrid),
  chosen over ParC-Net/MicroNet (both considered, both require external-repo integration since
  neither is in `timm`) specifically because it's natively `timm`-available. **Deliberately
  skipping a MobileViT-XS control run** on the same formulation (originally scoped to isolate the
  formulation-change variable) — explicit user choice, flagged as a methodological trade-off: if
  T3.1/StarNet-S1's result differs from T2.1's, it can't be attributed cleanly to the formulation
  change alone since the backbone changed too. Also note for the writeup: the offset-regression
  head is **not** part of the original PlaNet paper (PlaNet is pure cell classification) — it's
  an original addition, justified by analogy to object-detection class+offset prediction; cite
  PlaNet for the adaptive-cell idea only. **Implementation deferred to a future session on the
  Alex cluster** — this was a planning/discussion session only.
- Pretrained weights: **resolved 2026-08-17** — allowed, any source, as long as the
  submitted model is ≤5M params; fine-tuning on provided `train/` data is allowed. This
  reshapes Phase D candidate priorities (see `Experiment_Design.md` Tier 2+) — pretrained
  ImageNet backbones under 5M params (e.g. MobileNetV3-Small, EfficientNet-Lite0) are now
  in scope and should generally be preferred as a starting point over from-scratch training,
  since ~11.7k images is small for training a backbone from scratch.
- Mixup/cutmix-as-synthetic-data question — still open, send a short email before relying on
  it in augmentation policy.
- No deadline has been discussed yet — if there's a hard due date, worth laying a calendar
  over this flow so Phase D's loop has a concrete time budget instead of an open-ended one.
