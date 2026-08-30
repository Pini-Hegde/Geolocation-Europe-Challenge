"""How much of 59.22 is pretraining, how much is our training, how much is decoding?
Same whitening + local-mode decode applied to every row. Pooled OOF, official folds."""
import numpy as np, pandas as pd, json, torch
ROOT="/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"; BR=ROOT+"/best_result"; R=6371.0088
df=pd.read_csv(ROOT+"/geo_dataset/train_labels.csv"); lat,lng=df.lat.values,df.lng.values
fold=np.array([json.load(open(BR+"/folds/folds_official.json"))["filename_to_fold"][f] for f in df.filename])
def hav(a,b,c,d):
    a,b,c,d=map(np.radians,(a,b,c,d))
    h=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(h,0,1)))
def decode(Q,G,gi_lat,gi_lng,whiten,r,keep=192,pw=0.35):
    if whiten:
        mu=G.mean(0,keepdims=True); Gc=G-mu; C=Gc.T@Gc/len(Gc)
        w,V=np.linalg.eigh(C); o=np.argsort(-w); k=min(keep,len(w))
        w,V=w[o][:k],V[:,o][:,:k]; W=V@np.diag(np.maximum(w,1e-8)**(-pw))
        n=lambda X:X/np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-8)
        G,Q=n((G-mu)@W),n((Q-mu)@W)
    v,i=torch.from_numpy(Q@G.T).topk(32,dim=1); v=v.numpy(); ix=i.numpy()
    la,lo=gi_lat[ix],gi_lng[ix]
    if r==0: return la[:,0],lo[:,0]
    d1=hav(la[:,:1],lo[:,:1],la,lo); ww=(d1<r)*np.exp((v-v[:,:1])/0.05); ww/=ww.sum(1,keepdims=True)
    laR,loR=np.radians(la),np.radians(lo)
    x=(ww*np.cos(laR)*np.cos(loR)).sum(-1);y=(ww*np.cos(laR)*np.sin(loR)).sum(-1);z=(ww*np.sin(laR)).sum(-1)
    return np.degrees(np.arctan2(z,np.hypot(x,y))),np.degrees(np.arctan2(y,x))
def run(name,getfold):
    for wh,r,tag in [(False,0,"raw top-1"),(True,0,"+whitening"),(True,120,"+whitening +mode120")]:
        oof=np.zeros(len(df))
        for k in range(5):
            va,tr,Q,G=getfold(k)
            pa,po=decode(Q,G,lat[tr],lng[tr],wh,r)
            oof[va]=hav(lat[va],lng[va],pa,po)
        print(f"  {name:38s} {tag:22s} {np.median(oof):7.2f} km")
    print()
# A: ImageNet-pretrained ONLY, never trained on this data
E=np.load(ROOT+"/outputs/diag/emb_convnextv2_femto_448.npz",allow_pickle=True)["emb"].astype(np.float32)[:len(df)]
run("A: femto pretrained, NO training", lambda k:(np.where(fold==k)[0],np.where(fold!=k)[0],
    E[np.where(fold==k)[0]],E[np.where(fold!=k)[0]]))
# B: our trained model
def ours(k):
    z=np.load(f"{BR}/checkpoints/off_femto448_f{k}/emb.npz")
    return z["val_idx"],z["tr_idx"],z["val"].astype(np.float32),z["gal"].astype(np.float32)
run("B: femto AFTER our training", ours)
