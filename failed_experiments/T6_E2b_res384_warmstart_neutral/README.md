# T6_E2b — NEGATIVE: resolution 384 (warm-started) does not help

**Config:** warm-start from T6_E1 (K=4, 153.6 km) + input 288->384 + orientation. ONLY that
change. fold 0, 16 ep planned, killed at ep 5.

**Result:** median 163.8 / 169.5 / 165.5 / 167.9 over epochs 0/1/3/5 — flat, ~12-15 km WORSE
than the 288px E1 it started from, cell top-1 acc unchanged (0.42). No recovery trend.

**Read:** consistent with the earlier T5.1 384px continuation (also inconclusive/abandoned).
Bumping resolution on a 288-trained EfficientNet-B0 disrupts feature statistics and the
gain (if any) doesn't pay for it within a short fine-tune. Resolution is NOT the lever at
this model scale — the bottleneck is geo-cell granularity + classification accuracy
(see oracle analysis: perfect 96-cell classification = 85 km ceiling; model is at 150).

**Not ruled out:** training at 384 *from scratch* for a full schedule. Deprioritized —
finer cells + label smoothing is the higher-EV path.
