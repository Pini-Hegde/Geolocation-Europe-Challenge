"""Sanity checks that need no GPU: parameter budget of the shipped weights, and the
predictions.csv format. If the dataset is present, filenames are checked against
holdout_public/ too.

    python verify.py [--ckpt model/full_data/final.pt] [--pred predictions.csv]
"""
import os
import argparse
import pandas as pd
import torch
import geo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="model/full_data/final.pt")
    p.add_argument("--pred", default="predictions.csv")
    p.add_argument("--data", default=geo.DATA_ROOT)
    a = p.parse_args()

    net = geo.GeoNet(pretrained=False)
    net.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    n = net.n_params()
    for name, m in [("backbone", net.bb), ("projection", net.fc)]:
        print(f"  {name:12s} {sum(q.numel() for q in m.parameters()):>10,}")
    print(f"  {'gem p':12s} {net.p.numel():>10,}")
    print(f"params {n:,}  budget 5,000,000  compliant={n <= 5_000_000}")
    assert n <= 5_000_000

    d = pd.read_csv(a.pred)
    assert list(d.columns) == ["filename", "pred_lat", "pred_lng"], list(d.columns)
    assert len(d) == 2400, len(d)
    assert d.filename.is_unique and d.notna().all().all()
    assert d.pred_lat.between(-90, 90).all() and d.pred_lng.between(-180, 180).all()
    holdout = os.path.join(a.data, "holdout_public")
    if os.path.isdir(holdout):
        exp = sorted(f for f in os.listdir(holdout) if f.lower().endswith(".jpg"))
        assert sorted(d.filename) == exp, "filenames do not match holdout_public/"
        print("filenames match holdout_public/")
    print(f"predictions OK  {len(d)} rows  lat [{d.pred_lat.min():.2f}, {d.pred_lat.max():.2f}]"
          f"  lng [{d.pred_lng.min():.2f}, {d.pred_lng.max():.2f}]")


if __name__ == "__main__":
    main()
