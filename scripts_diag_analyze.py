"""Diagnostic: is holdout_public an IID (interleaved) sample of the same locations as train?
And what does pure image retrieval score under (a) an IID stratified split vs (b) the
project's 500m-group CV split?"""
import numpy as np, pandas as pd, json, sys
ROOT="/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"
E=np.load(sys.argv[1], allow_pickle=True)
emb=E["emb"].astype(np.float32); n_tr=int(E["n_train"])
df=pd.read_csv(ROOT+"/geo_dataset/train_labels.csv")
T=emb[:n_tr]; H=emb[n_tr:]
R=6371.0088
lat=df.lat.values; lng=df.lng.values
def hav(la1,lo1,la2,lo2):
    la1,lo1,la2,lo2=map(np.radians,(la1,lo1,la2,lo2))
    d=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(d,0,1)))
import torch
Tt=torch.from_numpy(T).cuda() if torch.cuda.is_available() else torch.from_numpy(T)
def topk(Q,G,k,exclude_self=False):
    Qt=torch.from_numpy(Q).to(Tt.device)
    S=Qt@G.T
    if exclude_self: S.fill_diagonal_(-2)
    v,i=S.topk(k,dim=1); return v.cpu().numpy(), i.cpu().numpy()

print("="*70); print("1) TRAIN->TRAIN top-1 (self excluded): sim vs true geo distance")
v,i=topk(T,Tt,1,exclude_self=True)
s1=v[:,0]; m1=i[:,0]
d1=hav(lat,lng,lat[m1],lng[m1])
print(f"  median top-1 sim {np.median(s1):.4f}")
print(f"  retrieval-as-prediction on TRAIN (leaky upper bound): median {np.median(d1):.2f} km, <25km {(d1<25).mean():.3f}")
print("\n  calibration  sim-bin -> P(match within 1/25/200 km), n:")
bins=np.quantile(s1,np.linspace(0,1,11))
for a,b in zip(bins[:-1],bins[1:]):
    m=(s1>=a)&(s1<b)
    if m.sum()==0: continue
    print(f"   [{a:.3f},{b:.3f}) n={m.sum():5d}  <1km {(d1[m]<1).mean():.3f}  <25km {(d1[m]<25).mean():.3f}  <200km {(d1[m]<200).mean():.3f}  med {np.median(d1[m]):8.1f}")

print("="*70); print("2) HOLDOUT->TRAIN top-1 similarity vs TRAIN->TRAIN top-1 similarity")
vh,ih=topk(H,Tt,1)
sh=vh[:,0]
print("  percentile |  train->train  |  holdout->train")
for p in [10,25,50,75,90,95,99]:
    print(f"    p{p:<3d}     |    {np.percentile(s1,p):.4f}     |    {np.percentile(sh,p):.4f}")
print(f"  frac above train p50 sim: holdout {(sh>np.median(s1)).mean():.3f} (0.5 if identical dists)")
# calibrated estimate of holdout oracle
thr=np.quantile(s1,np.linspace(0,1,21))
est={}
for lim in [1,25,200]:
    pr=[]
    for a,b in zip(thr[:-1],thr[1:]):
        m=(s1>=a)&(s1<b); pr.append((d1[m]<lim).mean() if m.sum() else 0)
    pr=np.array(pr)
    idx=np.clip(np.searchsorted(thr,sh)-1,0,len(pr)-1)
    est[lim]=pr[idx].mean()
print(f"\n  CALIBRATED estimate for holdout, using top-1 DINOv2 match as the prediction:")
print(f"    P(within 1 km)   ~ {est[1]:.3f}")
print(f"    P(within 25 km)  ~ {est[25]:.3f}")
print(f"    P(within 200 km) ~ {est[200]:.3f}")

print("="*70); print("3) HONEST retrieval score under two different validation splits")
cty=df.country.values
folds=json.load(open(ROOT+"/outputs/protocol/folds/folds.json"))
fmap=folds if isinstance(folds,dict) and "fold" not in folds else folds
try:
    fa=fmap["filename_to_fold"]
    fold=np.array([fa[f] for f in df.filename])
except Exception as e:
    print("  fold load fail",e, list(fmap)[:5]); fold=None

def eval_split(name, te_idx, tr_idx, k=1):
    G=Tt[tr_idx]
    v,i=topk(T[te_idx],G,min(k,len(tr_idx)))
    pi=tr_idx[i[:,0]]
    d=hav(lat[te_idx],lng[te_idx],lat[pi],lng[pi])
    print(f"  {name:34s} median {np.median(d):7.2f} km  mean {d.mean():7.1f}  <25km {(d<25).mean():.3f} <200km {(d<200).mean():.3f} <750km {(d<750).mean():.3f}")
    return d

rng=np.random.default_rng(0)
te=np.concatenate([rng.permutation(np.where(cty==c)[0])[:200] for c in np.unique(cty)])
mask=np.ones(len(T),bool); mask[te]=False; tr=np.where(mask)[0]
eval_split("IID 200/country split (=leaderboard?)", te, tr)
if fold is not None:
    te0=np.where(fold==0)[0]; tr0=np.where(fold!=0)[0]
    eval_split("project 500m-group CV fold 0", te0, tr0)
print("\n  (same retrieval method, same model - only the split differs)")
