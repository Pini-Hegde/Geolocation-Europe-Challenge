"""Tune the whitening + decode that produced the 41 km jump. Pooled OOF, official folds."""
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
def wh(G,Q,shrink,keep,pw):
    mu=G.mean(0,keepdims=True); Gc=G-mu
    C=Gc.T@Gc/len(Gc); C=(1-shrink)*C+shrink*np.eye(len(C))*np.trace(C)/len(C)
    w,V=np.linalg.eigh(C); o=np.argsort(-w); w,V=w[o][:keep],V[:,o][:,:keep]
    W=V@np.diag(np.maximum(w,1e-8)**(-pw))
    n=lambda X:X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-8)
    return n((G-mu)@W), n((Q-mu)@W)
tag=sys.argv[1]; D=int(sys.argv[2]) if len(sys.argv)>2 else 192
grid=list(itertools.product([0.0,0.1,0.3,0.5],[D,160,128,96],[0.5,0.35],[0,30,60,120]))
res={g:np.zeros(len(df)) for g in grid}; cov=np.zeros(len(df),bool)
for fp in sorted(glob.glob(f"{ROOT}/outputs/poc/{tag}*/emb.npz")):
    z=np.load(fp); va,tr=z["val_idx"],z["tr_idx"]
    Q0=z["val"].astype(np.float32); G0=z["gal"].astype(np.float32); cov[va]=True
    cache={}
    for sh,keep,pw,r in grid:
        if (sh,keep,pw) not in cache: cache[(sh,keep,pw)]=wh(G0,Q0,sh,keep,pw)
        G,Q=cache[(sh,keep,pw)]
        v,i=torch.from_numpy(Q@G.T).topk(32,dim=1); v=v.numpy(); gi=tr[i.numpy()]
        if r==0: pa,po=lat[gi[:,0]],lng[gi[:,0]]
        else:
            d1=hav(lat[gi[:,0]][:,None],lng[gi[:,0]][:,None],lat[gi],lng[gi])
            w=(d1<r)*np.exp((v-v[:,:1])/0.05); w/=w.sum(1,keepdims=True)
            pa,po=sph(lat[gi],lng[gi],w)
        res[(sh,keep,pw,r)][va]=hav(lat[va],lng[va],pa,po)
rows=sorted((np.median(v[cov]),g,v) for g,v in res.items())
print(f"coverage {cov.sum()}   [{tag}]")
print(f"{'shrink':>7} {'dim':>4} {'power':>6} {'r_km':>5} | {'MEDIAN':>8} {'mean':>7} {'<25km':>6} {'<200km':>7}")
for md,(sh,keep,pw,r),v in rows[:12]:
    d=v[cov]; print(f"{sh:7.2f} {keep:4d} {pw:6.2f} {r:5d} | {md:8.2f} {d.mean():7.1f} {(d<25).mean():6.3f} {(d<200).mean():7.3f}")
