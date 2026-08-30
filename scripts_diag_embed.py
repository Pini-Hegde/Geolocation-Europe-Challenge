"""Diagnostic ONLY (not part of the submitted model): embed train+holdout with DINOv2-S
to measure the dataset's near-duplicate / IID structure."""
import os, sys, numpy as np, pandas as pd, torch, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = "/home/hpc/v123be/v123be64/Public/Geolocation-Europe-Challenge"
D = ROOT + "/geo_dataset"
SIZE = int(os.environ.get("SIZE", 518))
MODEL = os.environ.get("MODEL", "vit_small_patch14_reg4_dinov2.lvd142m")
OUT = ROOT + f"/outputs/diag/emb_{MODEL.split('.')[0]}_{SIZE}.npz"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(D + "/train_labels.csv")
train_files = [D + "/train/" + f for f in df.filename]
hold_names = sorted(f for f in os.listdir(D + "/holdout_public") if f.lower().endswith(".jpg"))
hold_files = [D + "/holdout_public/" + f for f in hold_names]
files = train_files + hold_files
print(len(train_files), len(hold_files), flush=True)

m = timm.create_model(MODEL, pretrained=True, num_classes=0).eval().cuda().half()
cfg = timm.data.resolve_model_data_config(m); cfg["input_size"] = (3, SIZE, SIZE)
tf = timm.data.create_transform(**cfg, is_training=False)

class DS(Dataset):
    def __len__(self): return len(files)
    def __getitem__(self, i): return tf(Image.open(files[i]).convert("RGB"))

dl = DataLoader(DS(), batch_size=32, num_workers=12, pin_memory=True)
outs = []
with torch.no_grad():
    for bi, x in enumerate(dl):
        t = m.forward_features(x.cuda(non_blocking=True).half())
        npref = getattr(m, "num_prefix_tokens", 1)
        cls = t[:, 0]
        patch = t[:, npref:].mean(1)
        f = torch.cat([torch.nn.functional.normalize(cls.float(), dim=-1),
                       torch.nn.functional.normalize(patch.float(), dim=-1)], -1)
        outs.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float16))
        if bi % 50 == 0: print(bi, len(dl), flush=True)
E = np.concatenate(outs)
np.savez_compressed(OUT, emb=E, n_train=len(train_files), hold_names=np.array(hold_names))
print("saved", OUT, E.shape)
