"""Scores the pipeline on the held-out 20%.

Pass this the same seed trained with --val-frac 0.2 or your split gets re-built differently
 and you end up scoring test set pictures. Your gallery is the other 80% on which whitening and hubness were learned entirely.

Prints out the full ablation ladder, rather than a single number, so every stage has to prove its merit rather than claim it in the write-up.

There is no hyper-parameter search anywhere in this script, for clarity and to avoid its influence being claimed instead 
of the model's; everything below this line is what I considered a default setting.
"""
import argparse
import numpy as np
import geo
import embed


def report(name, d):
    print(f"{name:32s} median {np.median(d):7.2f}  mean {d.mean():8.2f}  "
          f"<25km {(d < 25).mean():.3f}  <200km {(d < 200).mean():.3f}  "
          f"<750km {(d < 750).mean():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True, help="npz from embed.py --split train")
    p.add_argument("--data", default=geo.DATA_ROOT)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csls-k", type=int, default=10)
    p.add_argument("--csls-beta", type=float, default=0.25)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--lam", type=float, default=3.0)
    p.add_argument("--radius", type=float, default=60.0)
    p.add_argument("--temp", type=float, default=0.1)
    a = p.parse_args()

    df = geo.load_labels(a.data)
    lat, lng = df.lat.values, df.lng.values
    gi, qi = geo.split_train_val(df, a.val_frac, a.seed)

    z = np.load(a.emb, allow_pickle=True)
    if list(z["names"]) != list(df.filename):
        raise SystemExit("embeddings must be in train_labels.csv order")
    X = embed.combine({s: z[f"emb{s}"] for s in embed.SIZES}, embed.SIZES)
    tokens = z["tokens"]
    print(f"gallery {len(gi)}  queries {len(qi)}")

    mu, Wm = geo.fit_whitening(X[gi])
    G, Q = geo.apply_whitening(X[gi], mu, Wm), geo.apply_whitening(X[qi], mu, Wm)
    glat, glng = lat[gi], lng[gi]

    # every row of the ladder decodes the same way; only the ranking in front of it changes
    def error(sim, idx):
        plat, plng = geo.local_mode_decode(sim, idx, glat, glng, a.radius, a.temp)
        return geo.haversine_km(lat[qi], lng[qi], plat, plng)

    sim, idx = geo.retrieve(Q, G, topk=a.topk)
    report("top-1 neighbour (whitened)", geo.haversine_km(lat[qi], lng[qi],
                                                         glat[idx[:, 0]], glng[idx[:, 0]]))
    report("+ local-mode decode", error(sim, idx))

    sim, idx = geo.retrieve(Q, G, geo.csls_hubness(G, a.csls_k), a.csls_beta, a.topk)
    report("+ CSLS hubness", error(sim, idx))

    sim, idx = geo.chamfer_rerank(tokens[qi], tokens[gi], idx, sim, a.lam)
    d = error(sim, idx)
    report("+ chamfer re-ranking", d)

    print("\nper country:")
    country = df.country.values[qi]
    for c in sorted(set(country), key=lambda c: -np.median(d[country == c])):
        e = d[country == c]
        print(f"  {c:16s} n={len(e):5d}  median {np.median(e):7.1f}  "
              f"<200km {(e < 200).mean():.3f}")


if __name__ == "__main__":
    main()
