# T6_E3c — finer cells (192/partition) with a plain one-hot loss = a wash vs E1

**Config:** E1's recipe, ONLY change = 192-cell partitions (K=3, geo_cells_multi_mid,
25-70 img/cell) instead of E1's 96-cell K=4. emb_dim 160 (to fit 5M). No EMA, no label
smoothing, orientation on. From ImageNet, 288px, 36 ep (killed at ep 25 on user request
before the final anneal epochs).

**Trajectory (median km, retrieval-mode decode):**
ep0 445 | ep5 234 | ep9 184 | ep11 168 | ep15 166 | ep19 168 | ep21 162 | ep23 164 | ep25 160
E1 for comparison: ep5 216 | ep9 188 | ep15 167 | ep23 164 | ep27 153.6 (best) then bounced ~157.

**Verdict:** the two curves overlap. Finer cells classify the coarse structure a bit faster
early, then plateau at the SAME place because the 192-way classifier caps out at
**cell top-1 acc 0.34** (vs E1's 0.43 at 96 cells) with heavy overfitting (train cell loss
-> 0.05, val acc flat). The precision gain of smaller cells is cancelled by the accuracy loss.

**Why (see [[geo-cell-accuracy-is-the-bottleneck]]):** the oracle 53 km ceiling for 192 cells
is only reachable with high classification accuracy. A one-hot loss punishes "adjacent cell"
(~60 km error) exactly as hard as "wrong country" (~1000 km), so the model hedges coarse.

**Not tested here (the actual levers):** haversine label smoothing (partial credit for
near-miss cells -> higher effective accuracy), and a decoder mode-radius tuned for finer
cells (E3c ran with r=250 km, way too wide).
