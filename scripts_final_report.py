import numpy as np, pandas as pd, json, glob, itertools, torch, timm, torch.nn as nn, torch.nn.functional as F
ROOT="/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"; R=6371.0088
df=pd.read_csv(ROOT+"/geo_dataset/train_labels.csv"); lat,lng,cty=df.lat.values,df.lng.values,df.country.values
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
grid=list(itertools.product([0.0,0.1,0.3],[192,160],[0.5,0.35,0.25],[0,60,120,200,300]))
err={g:np.full(len(df),np.nan) for g in grid}; raw=np.full(len(df),np.nan); foldof=np.full(len(df),-1)
for fp in sorted(glob.glob(f"{ROOT}/outputs/poc/off_femto448_f*/emb.npz")):
    k=int(fp.split("_f")[-1].split("/")[0]); z=np.load(fp)
    va,tr=z["val_idx"],z["tr_idx"]; foldof[va]=k
    Q0=z["val"].astype(np.float32); G0=z["gal"].astype(np.float32); cache={}
    i0=torch.from_numpy(Q0@G0.T).argmax(1).numpy(); raw[va]=hav(lat[va],lng[va],lat[tr[i0]],lng[tr[i0]])
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
cov=foldof>=0; oof=np.full(len(df),np.nan); picks={}
for k in range(5):
    tune=cov&(foldof!=k); g=min(grid,key=lambda g:np.nanmedian(err[g][tune])); picks[k]=g; oof[foldof==k]=err[g][foldof==k]
def stats(d): return dict(n=len(d),median=float(np.median(d)),mean=float(d.mean()),
    p25=float(np.percentile(d,25)),p75=float(np.percentile(d,75)),
    r25=float((d<25).mean()),r200=float((d<200).mean()),r750=float((d<750).mean()))
out={"pooled_raw":stats(raw[cov]),"pooled_final":stats(oof[cov]),
     "optimistic_same_data":float(min(np.nanmedian(v[cov]) for v in err.values())),
     "per_fold":{int(k):{"config":list(picks[k]),"raw":stats(raw[foldof==k]),"final":stats(oof[foldof==k])} for k in range(5)},
     "per_country":{c:{"n":int((cty==c).sum()),"raw":float(np.median(raw[cty==c])),"final":float(np.median(oof[cty==c])),
                       "r25":float((oof[cty==c]<25).mean()),"r200":float((oof[cty==c]<200).mean())} for c in sorted(set(cty))}}
# param breakdown
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.bb=timm.create_model("convnextv2_femto.fcmae_ft_in1k",pretrained=False,num_classes=0,global_pool="")
        s.p=nn.Parameter(torch.tensor(3.0)); s.fc=nn.Linear(s.bb.num_features,192)
n=Net(); out["params"]={"backbone":sum(p.numel() for p in n.bb.parameters()),"gem_p":1,
    "projection":sum(p.numel() for p in n.fc.parameters()),
    "whitening_mean":192,"whitening_matrix":192*192}
out["params"]["total"]=sum(out["params"].values())
json.dump(out,open(ROOT+"/outputs/final_report.json","w"),indent=2)
print(json.dumps(out,indent=1))
