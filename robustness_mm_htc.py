# robustness_mm_htc.py
import numpy as np, torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from dataset_video import VideoFaceDataset
from fft_feats import rgb_to_fft_amp
from model_mm_htc import MMHTC

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = "/Users/dhruvshrinet/Downloads/deepfake-detection-challenge/train_sample_videos/dataset_faces"
IMG_SIZE=224; T=8

val_tf = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
                             transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

def jpegify_tensor(img_t, q):
    img = (img_t*0.5+0.5).clamp(0,1)
    from io import BytesIO
    buf=BytesIO()
    Image.fromarray((img.permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(buf, "JPEG", quality=q)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
    return t(im)

@torch.no_grad()
def score_at_quality(model, ds, q):
    probs, tgts = [], []
    for x, y in ds:
        xq = torch.stack([jpegify_tensor(f, q) for f in x], dim=0).unsqueeze(0).to(DEVICE)
        xfft = rgb_to_fft_amp(xq.reshape(-1,*xq.shape[2:])).reshape(xq.shape)
        p = torch.sigmoid(model(xq, xfft)[0]).squeeze().item()
        probs.append(p); tgts.append(int(y))
    return roc_auc_score(tgts, probs)

if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    ds_val = VideoFaceDataset(ROOT, "val", T=T, transform=val_tf)
    model = MMHTC().to(DEVICE); model.load_state_dict(torch.load("mmhtc_best.pth", map_location=DEVICE)); model.eval()
    qualities = [10,20,40,60,80,95]
    aucs = [score_at_quality(model, ds_val, q) for q in qualities]
    plt.figure(figsize=(5,4)); plt.plot(qualities, aucs, marker='o')
    plt.xlabel("JPEG quality"); plt.ylabel("AUC"); plt.title("Robustness (val)")
    plt.tight_layout(); plt.savefig("figures/robustness_curves.pdf")
    print("Saved figures/robustness_curves.pdf")
