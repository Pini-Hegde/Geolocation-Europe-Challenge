"""Adversarial checks on the 59.22 km result."""
import numpy as np, pandas as pd, json, glob, torch
ROOT="/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"; BR=ROOT+"/best_result"; R=6371.0088
df=pd.read_csv(ROOT+"/geo_dataset/train_labels.csv"); lat,lng=df.lat.values,df.lng.values
off=json.load(open(BR+"/folds/folds_official.json"))["filename_to_fold"]
fold=np.array([off[f] for f in df.filename])
def hav(a,b,c,d):
    a,b,c,d=map(np.radians,(a,b,c,d))
    h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))
print("CHECK 1 — train/val index integrity per fold")
ok=True
for k in range(5):
    z=np.load(f"{BR}/checkpoints/off_femto448_f{k}/emb.npz")
    va,tr=z["val_idx"],z["tr_idx"]
    ov=len(set(va)&set(tr)); exp_va=set(np.where(fold==k)[0]); exp_tr=set(np.where(fold!=k)[0])
    good = ov==0 and set(va)==exp_va and set(tr)==exp_tr and len(va)+len(tr)==len(df)
    ok &= good
    print(f"  fold {k}: |val|={len(va)} |gal|={len(tr)} overlap={ov} matches_official={set(va)==exp_va} -> {'OK' if good else 'FAIL'}")
print(f"  => {'no train/val contamination' if ok else 'CONTAMINATION'}\n")

print("CHECK 2 — negative control: shuffle the gallery's coordinates")
print("  (if the pipeline still scores well, the result is an artefact)")
def decode(Q,G,tr,va,glat,glng,shrink=0.0,keep=192,pw=0.35,r=120):
    mu=G.mean(0,keepdims=True); Gc=G-mu; C=Gc.T@Gc/len(Gc)
    C=(1-shrink)*C+shrink*np.eye(len(C))*np.trace(C)/len(C)
    w,V=np.linalg.eigh(C); o=np.argsort(-w); w,V=w[o][:keep],V[:,o][:,:keep]
    W=V@np.diag(np.maximum(w,1e-8)**(-pw))
    n=lambda X:X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-8)
    G2,Q2=n((G-mu)@W),n((Q-mu)@W)
    v,i=torch.from_numpy(Q2@G2.T).topk(32,dim=1); v=v.numpy(); ix=i.numpy()
    la,lo=glat[ix],glng[ix]
    d1=hav(la[:,:1],lo[:,:1],la,lo); ww=(d1<r)*np.exp((v-v[:,:1])/0.05); ww/=ww.sum(1,keepdims=True)
    laR,loR=np.radians(la),np.radians(lo)
    x=(ww*np.cos(laR)*np.cos(loR)).sum(-1);y=(ww*np.cos(laR)*np.sin(loR)).sum(-1);z2=(ww*np.sin(laR)).sum(-1)
    return np.degrees(np.arctan2(z2,np.hypot(x,y))),np.degrees(np.arctan2(y,x))
rng=np.random.default_rng(0)
for label,shuffle in [("real coordinates",False),("SHUFFLED gallery coords",True)]:
    oof=np.zeros(len(df))
    for k in range(5):
        z=np.load(f"{BR}/checkpoints/off_femto448_f{k}/emb.npz")
        va,tr=z["val_idx"],z["tr_idx"]; Q=z["val"].astype(np.float32); G=z["gal"].astype(np.float32)
        gl,gn=lat[tr].copy(),lng[tr].copy()
        if shuffle:
            p=rng.permutation(len(tr)); gl,gn=gl[p],gn[p]
        pa,po=decode(Q,G,tr,va,gl,gn)
        oof[va]=hav(lat[va],lng[va],pa,po)
    print(f"  {label:26s} median {np.median(oof):7.2f} km")
