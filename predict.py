"""Writes predictions.csv for holdout_public/.

    gallery     all 11,758 labelled training images, embedded with multi-scale TTA
    ranking     cosine after gallery-fitted PCA-whitening, minus the CSLS hubness term
    re-ranking  top-64 shortlist re-scored by chamfer similarity over spatial tokens
    decode      similarity-weighted spherical mean of whatever survives within RADIUS km
                of the new top-1

Whitening and hubness statistics are fitted on the gallery only, and every query is
decoded independently of every other query (no transductive tricks).

The decode hyper-parameters below were fixed on the validation split and are shared with
eval.py. Runs on CPU if no GPU is available.
"""
import argparse
import numpy as np
import pandas as pd
import geo
import embed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gallery", required=True, help="npz from embed.py --split train")
    p.add_argument("--query", required=True, help="npz from embed.py --split holdout")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--data", default=geo.DATA_ROOT)
    p.add_argument("--csls-k", type=int, default=10)
    p.add_argument("--csls-beta", type=float, default=0.25)
    p.add_argument("--topk", type=int, default=64, help="shortlist size before re-ranking")
    p.add_argument("--lam", type=float, default=3.0, help="weight of the chamfer term")
    p.add_argument("--radius", type=float, default=60.0)
    p.add_argument("--temp", type=float, default=0.1)
    a = p.parse_args()

    df = geo.load_labels(a.data)
    zg = np.load(a.gallery, allow_pickle=True)
    zq = np.load(a.query, allow_pickle=True)
    if list(zg["names"]) != list(df.filename):
        raise SystemExit("gallery embeddings must be in train_labels.csv order")

    G = embed.combine({s: zg[f"emb{s}"] for s in embed.SIZES}, embed.SIZES)
    Q = embed.combine({s: zq[f"emb{s}"] for s in embed.SIZES}, embed.SIZES)
    mu, Wm = geo.fit_whitening(G)
    G, Q = geo.apply_whitening(G, mu, Wm), geo.apply_whitening(Q, mu, Wm)

    sim, idx = geo.retrieve(Q, G, geo.csls_hubness(G, a.csls_k), a.csls_beta, a.topk)
    sim, idx = geo.chamfer_rerank(zq["tokens"], zg["tokens"], idx, sim, a.lam)
    lat, lng = geo.local_mode_decode(sim, idx, df.lat.values, df.lng.values,
                                     a.radius, a.temp)

    out = pd.DataFrame({"filename": [str(n) for n in zq["names"]],
                        "pred_lat": lat, "pred_lng": lng})
    out.to_csv(a.out, index=False)
    print(out.head().to_string(index=False))
    print(f"wrote {a.out}: {len(out)} rows")


if __name__ == "__main__":
    main()
