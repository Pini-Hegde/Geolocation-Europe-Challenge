"""Pooled-OOF decoder sweep over saved per-fold retrieval embeddings.
Tries PCA-whitening, alpha query expansion, and local-mode decoding. No retraining."""
import numpy as np, pandas as pd, json, sys, glob, itertools, torch
ROOT="/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"; R=6371.0088
df=pd.read_csv(ROOT+"/geo_dataset/train_labels.csv"); lat,lng=df.lat.values,df.lng.values
def hav(a,b,c,d):
    a,b,c,d=map(np.radians,(a,b,c,d))
    h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))
def sph(la,lo,w):
    la,lo=np.radians(la),np.radians(lo)
    x=(w*np.cos(la)*np.cos(lo)).sum(-1); y=(w*np.cos(la)*np.sin(lo)).sum(-1); z=(w*np.sin(la)).sum(-1)
    return np.degrees(np.arctan2(z,np.hypot(x,y))), np.degrees(np.arctan2(y,x))
def whiten(G,Q,eps=1e-3,shrink=0.5):
    mu=G.mean(0,keepdims=True); Gc=G-mu
    C=(Gc.T@Gc)/len(Gc); C=(1-shrink)*C+shrink*np.eye(len(C))*np.trace(C)/len(C)
    w,V=np.linalg.eigh(C); w=np.maximum(w,eps)
    W=V@np.diag(w**-0.5)@V.T
    f=lambda X:(X-mu)@W
    n=lambda X:X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-8)
    return n(f(G)),n(f(Q))
def aqe(G,Q,k=3,alpha=3.0):
    S=Q@G.T; v,i=torch.topk(torch.from_numpy(S),k,dim=1); v=v.numpy(); i=i.numpy()
    w=np.clip(v,0,None)**alpha; w/=w.sum(1,keepdims=True)
    Qe=Q+ (w[:,:,None]*G[i]).sum(1)
    return Qe/np.maximum(np.linalg.norm(Qe,axis=1,keepdims=True),1e-8)
tags=sys.argv[1:]
cfgs=list(itertools.product([False,True],[False,True],["top1","mode30","mode60"]))
res={c:np.zeros(len(df)) for c in cfgs}; cov=np.zeros(len(df),bool)
for t in tags:
    for fp in sorted(glob.glob(f"{ROOT}/outputs/poc/{t}*/emb.npz")):
        z=np.load(fp); va,tr=z["val_idx"],z["tr_idx"]
        Q0=z["val"].astype(np.float32); G0=z["gal"].astype(np.float32); cov[va]=True
        for wh,qe,dec in cfgs:
            G,Q=(whiten(G0,Q0) if wh else (G0,Q0))
            if qe: Q=aqe(G,Q)
            S=torch.from_numpy(Q@G.T); v,i=S.topk(32,dim=1); v=v.numpy(); gi=tr[i.numpy()]
            if dec=="top1": pa,po=lat[gi[:,0]],lng[gi[:,0]]
            else:
                r=int(dec[4:])
                d1=hav(lat[gi[:,0]][:,None],lng[gi[:,0]][:,None],lat[gi],lng[gi])
                w=(d1<r)*np.exp((v-v[:,:1])/0.05); w/=w.sum(1,keepdims=True)
                pa,po=sph(lat[gi],lng[gi],w)
            res[(wh,qe,dec)][va]=hav(lat[va],lng[va],pa,po)
n=cov.sum(); print(f"coverage {n}/{len(df)} images\n")
rows=sorted(((np.median(v[cov]),k,v) for k,v in res.items()))
print(f"{'whiten':>7} {'aQE':>5} {'decode':>7} | {'MEDIAN':>8} {'mean':>7} {'<25km':>6} {'<200km':>7} {'<750km':>7}")
for md,(wh,qe,dec),v in rows:
    d=v[cov]
    print(f"{str(wh):>7} {str(qe):>5} {dec:>7} | {md:8.2f} {d.mean():7.1f} {(d<25).mean():6.3f} {(d<200).mean():7.3f} {(d<750).mean():7.3f}")
