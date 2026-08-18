# EDA baselines B0 / B1 — summary

Group-stratified 5-fold CV (`outputs/protocol/folds/folds.json`). Centroids are spherical means (unit-vector average, converted back), not naive degree averages.

## Headline (mean ± std across folds)
| baseline | median km | mean km | < 200km (city) | < 750km (country) |
|---|---:|---:|---:|---:|
| B0 — global mean | 1217.8 ± 9.1 | 1324.3 | 2.3% | 20.1% |
| B1 — oracle country centroid | 288.1 ± 4.2 | 333.1 | 28.0% | 96.9% |

## Reading B0 vs. B1
B1 assumes the true country is already known (oracle, upper-bound only) — the gap between B0 and B1 is what perfect country classification alone would buy. What's left in B1's own error is purely within-country imprecision, which no amount of country-level classification can fix — it lower-bounds what any grid-cell / classification-heavy approach could achieve without also getting finer localization right.

## B1 per-country breakdown (pooled across folds)
Countries with the highest remaining median error, even given the true country, are where within-country precision matters most (see `b1_per_country.csv` for the full table):

| country | n | median km | mean km | < 200km |
|---|---:|---:|---:|---:|
| Norway | 1000 | 520.9 | 503.3 | 11.0% |
| Spain | 1000 | 468.2 | 557.5 | 13.8% |
| Turkey | 1000 | 408.1 | 417.5 | 14.1% |
| Italy | 1000 | 383.9 | 366.2 | 15.1% |
| Sweden | 1000 | 350.6 | 376.4 | 18.8% |
| France | 1000 | 339.0 | 348.1 | 18.2% |
| United_Kingdom | 1000 | 307.6 | 314.2 | 23.9% |
| Finland | 1000 | 259.1 | 261.8 | 31.4% |
| Germany | 1000 | 254.5 | 247.7 | 30.9% |
| Poland | 934 | 229.0 | 222.1 | 38.5% |
| Belarus | 864 | 192.0 | 181.4 | 57.3% |
| Iceland | 960 | 172.5 | 166.6 | 69.2% |

*(see `b1_per_country.csv` for all countries)*
