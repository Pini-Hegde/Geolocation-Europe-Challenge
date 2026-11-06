# One target per pipeline step. Override paths, e.g.  make predict GEO_DATA=/data/geo_dataset
GEO_DATA ?= geo_dataset
CKPT     ?= model/full_data/final.pt
PY       ?= python3
export GEO_DATA

.PHONY: install predict train validate verify

install:            ## pinned CUDA 12.6 environment
	$(PY) -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

predict:            ## shipped weights -> predictions.csv (~7 min, deterministic)
	$(PY) embed.py --ckpt $(CKPT) --split train   --out gallery.npz
	$(PY) embed.py --ckpt $(CKPT) --split holdout --out holdout.npz
	$(PY) predict.py --gallery gallery.npz --query holdout.npz --out predictions.csv

train:              ## retrain on all 11,758 images (~67 min); then: make predict CKPT=model/retrain/final.pt
	$(PY) train.py --out model/retrain --epochs 40

validate:           ## train on 80%, score the held-out 20% with the full ablation ladder
	$(PY) train.py --out model/val --val-frac 0.2 --epochs 40
	$(PY) embed.py --ckpt model/val/final.pt --split train --out val_emb.npz
	$(PY) eval.py --emb val_emb.npz

verify:             ## parameter budget + predictions.csv format (CPU only)
	$(PY) verify.py --ckpt $(CKPT)
