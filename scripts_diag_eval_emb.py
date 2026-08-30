"""Score any embedding file as a pure geo-retrieval predictor, under both splits."""
import numpy as np, pandas as pd, json, sys, torch, glob, os
ROOT="/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"
df=pd.read_csv(ROOT+"/geo_dataset/train_labels.csv"); R=6371.0088
lat,lng,cty=df.lat.values,df.lng.values,df.country.values
fold=np.array([json.load(open(ROOT+"/outputs/protocol/folds/folds_official.json"))["filename_to_fold"][f] for f in df.filename])
def hav(a,b,c,d):
    a,b,c,d=map(np.radians,(a,b,c,d))
    h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))
def sph_mean(la,lo,w):
    la,lo=np.radians(la),np.radians(lo)
    x=(w*np.cos(la)*np.cos(lo)).sum(-1); y=(w*np.cos(la)*np.sin(lo)).sum(-1); z=(w*np.sin(la)).sum(-1)
    return np.degrees(np.arctan2(z,np.hypot(x,y))), np.degrees(np.arctan2(y,x))
SPLITS={f"official fold {k}":(np.where(fold==k)[0],np.where(fold!=k)[0]) for k in range(5)}
def report(tag,d): print(f"   {tag:28s} med {np.median(d):7.2f}  mean {d.mean():7.1f}  <25 {(d<25).mean():.3f} <200 {(d<200).mean():.3f} <750 {(d<750).mean():.3f}")
for path in sorted(sys.argv[1:]):
    E=np.load(path,allow_pickle=True); T=torch.from_numpy(E["emb"].astype(np.float32)[:len(df)]).cuda()
    print("\n"+"="*78); print(os.path.basename(path), T.shape)
    for sname,(te,tr) in SPLITS.items():
        S=T[te]@T[tr].T
        v,i=S.topk(32,dim=1); v=v.cpu().numpy(); gi=tr[i.cpu().numpy()]
        print(f"  [{sname}]  n_test={len(te)}")
        report("top-1 NN",hav(lat[te],lng[te],lat[gi[:,0]],lng[gi[:,0]]))
        for k,tau in [(8,0.03),(16,0.05)]:
            w=np.exp((v[:,:k]-v[:,:1])/tau); w/=w.sum(1,keepdims=True)
            pa,po=sph_mean(lat[gi[:,:k]],lng[gi[:,:k]],w); report(f"softmax-{k} sph.mean tau{tau}",hav(lat[te],lng[te],pa,po))
        # local mode: among top-32, keep those within 30km of top-1, weighted mean
        d1=hav(lat[gi[:,0]][:,None],lng[gi[:,0]][:,None],lat[gi],lng[gi])
        w=(d1<30)*np.exp((v-v[:,:1])/0.05); w/=w.sum(1,keepdims=True)
        pa,po=sph_mean(lat[gi],lng[gi],w); report("top32 local-mode r30km",hav(lat[te],lng[te],pa,po))
    # pooled OOF over all 5 official folds
    oof=np.zeros(len(df)); 
    for k in range(5):
        te,tr=np.where(fold==k)[0],np.where(fold!=k)[0]
        i=(T[te]@T[tr].T).argmax(1).cpu().numpy(); gi=tr[i]
        oof[te]=hav(lat[te],lng[te],lat[gi],lng[gi])
    print(f"  >>> POOLED OOF top-1: median {np.median(oof):7.2f} km  mean {oof.mean():7.1f} "
          f"<200 {(oof<200).mean():.3f} <750 {(oof<750).mean():.3f}")
    del T; torch.cuda.empty_cache()
