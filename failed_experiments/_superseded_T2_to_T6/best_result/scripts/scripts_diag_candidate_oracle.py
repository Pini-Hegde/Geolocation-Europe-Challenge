"""Is the bottleneck the REPRESENTATION (right answer not in the shortlist) or the
HEAD/DECODER (it's there but we rank it badly)? Pooled OOF over official folds."""
import numpy as np, pandas as pd, json, sys, glob, torch
ROOT="/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"; BR=ROOT+"/best_result"; R=6371.0088
df=pd.read_csv(ROOT+"/geo_dataset/train_labels.csv"); lat,lng=df.lat.values,df.lng.values
fold=np.array([json.load(open(BR+"/folds/folds_official.json"))["filename_to_fold"][f] for f in df.filename])
def hav(a,b,c,d):
    a,b,c,d=map(np.radians,(a,b,c,d))
    h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))
def report(name,Q,G,va,tr,K=(1,5,10,32,100)):
    S=torch.from_numpy(Q@G.T).cuda(); v,i=S.topk(max(K),dim=1); gi=tr[i.cpu().numpy()]
    d=hav(lat[va][:,None],lng[va][:,None],lat[gi],lng[gi])
    return {k:d[:,:k].min(1) for k in K}, d[:,0]
def run(name, get):
    acc={}; top1=np.zeros(len(df)); cov=np.zeros(len(df),bool)
    for k in range(5):
        va,tr,Q,G=get(k)
        if va is None: continue
        o,t1=report(name,Q,G,va,tr); top1[va]=t1; cov[va]=True
        for kk,v in o.items(): acc.setdefault(kk,np.zeros(len(df)))[va]=v
    n=cov.sum()
    print(f"\n{name}  (n={n})")
    print(f"   top-1 (what we actually score) : median {np.median(top1[cov]):7.2f} km")
    for kk in sorted(acc):
        if kk==1: continue
        print(f"   ORACLE best-of-top-{kk:<4d}          : median {np.median(acc[kk][cov]):7.2f} km   "
              f"<25km {(acc[kk][cov]<25).mean():.3f}")
# DINOv2 (ineligible, upper reference)
E=np.load(ROOT+"/outputs/diag/emb_vit_small_patch14_reg4_dinov2_518.npz",allow_pickle=True)["emb"].astype(np.float32)[:len(df)]
run("DINOv2-S 22M (ineligible reference)", lambda k:(np.where(fold==k)[0],np.where(fold!=k)[0],
     E[np.where(fold==k)[0]],E[np.where(fold!=k)[0]]))
# our trained femto folds
def femto(k):
    p=glob.glob(f"{BR}/checkpoints/off_femto448_f{k}/emb.npz")
    if not p: return None,None,None,None
    z=np.load(p[0]); return z["val_idx"],z["tr_idx"],z["val"].astype(np.float32),z["gal"].astype(np.float32)
run("convnextv2_femto 4.9M (ours, trained)", femto)
