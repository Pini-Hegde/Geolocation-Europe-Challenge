"""Cross-fitted decoder tuning: pick the config on the OTHER 4 folds, apply to the held-out
fold, pool. Removes the selection bias of tuning and reporting on the same data."""
import numpy as np, pandas as pd, json, sys, glob, itertools, torch
ROOT="/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"; BR=ROOT+"/best_result"; R=6371.0088
df=pd.read_csv(ROOT+"/geo_dataset/train_labels.csv"); lat,lng=df.lat.values,df.lng.values
def hav(a,b,c,d):
    a,b,c,d=map(np.radians,(a,b,c,d))
    h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))
def sph(la,lo,w):
    la,lo=np.radians(la),np.radians(lo)
    x=(w*np.cos(la)*np.cos(lo)).sum(-1);y=(w*np.cos(la)*np.sin(lo)).sum(-1);z=(w*np.sin(la)).sum(-1)
    return np.degrees(np.arctan2(z,np.hypot(x,y))),np.degrees(np.arctan2(y,x))
def wh(G,Q,shrink,keep,pw):
    mu=G.mean(0,keepdims=True); Gc=G-mu
    C=Gc.T@Gc/len(Gc); C=(1-shrink)*C+shrink*np.eye(len(C))*np.trace(C)/len(C)
    w,V=np.linalg.eigh(C); o=np.argsort(-w); w,V=w[o][:keep],V[:,o][:,:keep]
    W=V@np.diag(np.maximum(w,1e-8)**(-pw))
    n=lambda X:X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-8)
    return n((G-mu)@W),n((Q-mu)@W)
tags=sys.argv[1:]
grid=list(itertools.product([0.0,0.1,0.3],[192,160],[0.5,0.35,0.25],[0,60,120,200,300]))
err={g:np.full(len(df),np.nan) for g in grid}
foldof=np.full(len(df),-1)
for t in tags:
    for fp in sorted(glob.glob(f"{BR}/checkpoints/{t}*/emb.npz")):
        k=int(fp.split("_f")[-1].split("/")[0]); z=np.load(fp)
        va,tr=z["val_idx"],z["tr_idx"]; foldof[va]=k
        Q0=z["val"].astype(np.float32); G0=z["gal"].astype(np.float32); cache={}
        for sh,keep,pw,r in grid:
            if (sh,keep,pw) not in cache: cache[(sh,keep,pw)]=wh(G0,Q0,sh,keep,pw)
            G,Q=cache[(sh,keep,pw)]
            v,i=torch.from_numpy(Q@G.T).topk(32,dim=1); v=v.numpy(); gi=tr[i.numpy()]
            if r==0: pa,po=lat[gi[:,0]],lng[gi[:,0]]
            else:
                d1=hav(lat[gi[:,0]][:,None],lng[gi[:,0]][:,None],lat[gi],lng[gi])
                w=(d1<r)*np.exp((v-v[:,:1])/0.05); w/=w.sum(1,keepdims=True)
                pa,po=sph(lat[gi],lng[gi],w)
            err[(sh,keep,pw,r)][va]=hav(lat[va],lng[va],pa,po)
cov=foldof>=0
best_oracle=min(np.nanmedian(v[cov]) for v in err.values())
oof=np.full(len(df),np.nan); picks={}
for k in range(5):
    tune=cov&(foldof!=k); ev=foldof==k
    g=min(grid,key=lambda g:np.nanmedian(err[g][tune])); picks[k]=g
    oof[ev]=err[g][ev]
d=oof[cov]
print(f"n={cov.sum()}   configs={len(grid)}")
print(f"  OPTIMISTIC (tuned & reported on same data) : {best_oracle:6.2f} km")
print(f"  HONEST     (cross-fitted decoder)          : {np.nanmedian(d):6.2f} km")
print(f"     mean {np.nanmean(d):7.1f}  <25km {(d<25).mean():.3f}  <200km {(d<200).mean():.3f}  <750km {(d<750).mean():.3f}")
print("  per-fold config picked (shrink,dim,power,radius):")
for k,g in picks.items(): print(f"    fold {k}: {g}")
