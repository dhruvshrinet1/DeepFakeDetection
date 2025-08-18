# dataset_video.py
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VideoFaceDataset(Dataset):
    def __init__(self, root, split, T=8, transform=None):
        self.root = Path(root)/split
        self.T = T
        self.transform = transform
        self.samples = []  # (list_of_frame_paths, label)
        for lab in ["REAL","FAKE"]:
            for vid_dir in sorted((self.root/lab).glob("*")):
                frames = sorted(vid_dir.glob("*.jpg"))
                if len(frames) >= 2:
                    self.samples.append((frames, 0 if lab=="REAL" else 1))
        self.classes = ["REAL","FAKE"]

    def _pick_indices(self, n):
        t = min(self.T, n)
        idx = np.linspace(0, n-1, num=t, dtype=int).tolist()
        # pad by repeating last if needed
        while len(idx) < self.T: idx.append(idx[-1])
        return idx

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        frames, y = self.samples[i]
        idxs = self._pick_indices(len(frames))
        imgs = []
        for k in idxs:
            img = Image.open(frames[k]).convert("RGB")
            img = self.transform(img) if self.transform else transforms.ToTensor()(img)
            imgs.append(img)
        x = torch.stack(imgs, dim=0)  # [T,3,224,224]
        return x, torch.tensor(y).long()
