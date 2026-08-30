# Submission candidates

## preds_kdtree_fulldata_mode_noTTA.csv  ← current best, ready to upload
- Model: T6_FULL_s1 — the T6 recipe (EfficientNet-B0, K=4 x 384 haversine-smoothed geo-cells,
  shared low-rank cell trunk, 224-d contrastive embedding, orientation correction) trained on
  **100% of the labelled data**, 55 epochs. 4,944,970 params (compliant, single model).
- Decode: mode-seeking spherical-mean retrieval, retrieval-weight 16, cluster radius 200 km,
  top-m 100, hard kernel. NO test-time augmentation (flip inverts driving-side/signage cues
  and hurt by ~9 km on CV).
- Expected holdout median ≈ 122–128 km  (E3e fold-0 CV of this decode = 132.7;
  full-data trains on ~25% more data; historical CV→holdout gap ≈ −11 km).
- vs current submitted model: 139.8 km (rank #5). This should reach ~#4.

To regenerate:
    python -m src.predict_ensemble \
      --spec outputs/experiments/T6_FULL_s1/fold-1/best_model.pt:geo_cells_multi_fine \
      --decoder mode --retrieval-weight 16 --radius-km 200 --top-m 100 --kernel hard \
      --output submission_candidates/preds_kdtree_fulldata_mode_noTTA.csv
