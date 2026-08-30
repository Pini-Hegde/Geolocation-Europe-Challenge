# T6_E2 — FAILED: label smoothing bolted onto a warm-started model

**Config:** warm-start from T6_E1 (K=4, 153.6 km fold-0) + input 384 + PIGEON haversine
label smoothing (tau=100) + EMA(0.999) + contrastive-country-blend(0.35). fold 0, 18 ep.

**Result:** median regressed 153.6 -> ~170 km. Cell top-1 accuracy collapsed 0.42 -> 0.35
over the first ~5 epochs and never recovered. Best epoch was epoch 0 (158.8 km), i.e.
*before* the smoothed objective took effect.

**Why it failed:**
- Switching the cell-classifier target from one-hot to a haversine-Gaussian soft label
  *mid-training* makes the warm-started (already-sharp) model un-learn its cell predictions.
- tau=100 km is too aggressive for this project's small within-country cells
  (~40-120 images/cell, centroids often <100 km apart) — the soft target is nearly uniform
  over a country's cells.
- 4 changes bundled -> resolution-384's effect (if any) is unmeasurable here.

**Follow-ups:** test res-384 alone (warm-start, no objective change); if label smoothing is
revisited it must be from-scratch with tau >= 250, or applied only as a small auxiliary term.
