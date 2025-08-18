# python - <<'PY'
import os, shutil, random, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC = Path("/Users/dhruvshrinet/Downloads/deepfake-detection-challenge/train_sample_videos/output_path")
DST = Path("/Users/dhruvshrinet/Downloads/deepfake-detection-challenge/train_sample_videos/dataset_faces")
for sub in ["train/REAL","train/FAKE","val/REAL","val/FAKE"]:
    (DST/sub).mkdir(parents=True, exist_ok=True)

samples = []
for lab in ["REAL","FAKE"]:
    for vid in sorted((SRC/lab).glob("*")):
        samples.append((vid, 0 if lab=="REAL" else 1))

idx = np.arange(len(samples))
labels = np.array([y for _,y in samples])
tr, va = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)

def link(vpath, split):
    lab = "REAL" if vpath.parent.name=="REAL" else "FAKE"
    dst = DST/split/lab/vpath.name
    if dst.exists(): return
    dst.mkdir(parents=True, exist_ok=True)
    for img in vpath.glob("*.jpg"):
        os.link(img, dst/img.name)  # hardlink; use os.symlink on mac if you prefer
for i in tr: link(samples[i][0], "train")
for i in va: link(samples[i][0], "val")
print("done. dataset at:", DST)
# PY
