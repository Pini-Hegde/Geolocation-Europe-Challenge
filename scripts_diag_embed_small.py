"""Embed train+holdout with several <=5M-param ImageNet backbones, GeM-pooled, for the
retrieval-ceiling diagnostic."""
import os, numpy as np, pandas as pd, torch, timm, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
ROOT="/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"; D=ROOT+"/geo_dataset"
df=pd.read_csv(D+"/train_labels.csv")
hold=sorted(f for f in os.listdir(D+"/holdout_public") if f.lower().endswith(".jpg"))
files=[D+"/train/"+f for f in df.filename]+[D+"/holdout_public/"+f for f in hold]
os.makedirs(ROOT+"/outputs/diag",exist_ok=True)
MODELS=os.environ["MODELS"].split(","); SIZE=int(os.environ.get("SIZE",448))
for name in MODELS:
    out=f"{ROOT}/outputs/diag/emb_{name.split('.')[0]}_{SIZE}.npz"
    if os.path.exists(out): print("skip",out); continue
    m=timm.create_model(name,pretrained=True,num_classes=0,global_pool="").eval().cuda().half()
    n=sum(p.numel() for p in m.parameters())/1e6
    cfg=timm.data.resolve_model_data_config(m); cfg["input_size"]=(3,SIZE,SIZE); cfg["crop_pct"]=1.0
    tf=timm.data.create_transform(**cfg,is_training=False)
    class DS(Dataset):
        def __len__(s): return len(files)
        def __getitem__(s,i): return tf(Image.open(files[i]).convert("RGB"))
    dl=DataLoader(DS(),batch_size=32,num_workers=12,pin_memory=True)
    outs=[]
    with torch.no_grad():
        for x in dl:
            f=m(x.cuda(non_blocking=True).half()).float()
            if f.ndim==3: f=f.transpose(1,2)                      # B,C,N (vit)
            elif f.ndim==4: f=f.flatten(2)                        # B,C,HW (cnn)
            gem=(f.clamp(min=1e-6).pow(3).mean(-1)).pow(1/3)      # GeM p=3
            avg=f.mean(-1)
            v=torch.cat([F.normalize(gem,dim=-1),F.normalize(avg,dim=-1)],-1)
            outs.append(F.normalize(v,dim=-1).cpu().numpy().astype(np.float16))
    E=np.concatenate(outs)
    np.savez_compressed(out,emb=E,n_train=len(df),hold_names=np.array(hold))
    print(f"DONE {name} {n:.2f}M -> {out} {E.shape}",flush=True)
    del m; torch.cuda.empty_cache()
