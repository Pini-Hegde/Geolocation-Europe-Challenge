# T6_E3f — country-weight 2.5 + contrastive-country-blend 0.6 does NOT lift country accuracy

E3e recipe + those 2 changes. Killed ep5: country top-1 acc 0.744 == E3e (0.745) same epoch,
but median 241 vs E3e 214 (27km worse). Country acc in E3e barely moves from ~0.74 across all
55 epochs -> the ~0.78 country ceiling is a capacity/feature-quality limit (single Linear head
+ EffNet-B0 can't separate rural FR/DE/IT), not a loss-weight problem. Next: MLP country head.
