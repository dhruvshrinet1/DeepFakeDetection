# calibrate_mm_htc.py
import torch, numpy as np
from pathlib import Path
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from dataset_video import VideoFaceDataset
from fft_feats import rgb_to_fft_amp
from model_mm_htc import MMHTC
from torchvision import transforms

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = "/Users/dhruvshrinet/Downloads/deepfake-detection-challenge/train_sample_videos/dataset_faces"
IMG_SIZE=224; T=8
val_tf = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
                             transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])

@torch.no_grad()
def logits_labels(model, ds):
    model.eval()
    L, Y = [], []
    for x,y in ds:
        x = x.unsqueeze(0).to(DEVICE)
        xfft = rgb_to_fft_amp(x.reshape(-1,*x.shape[2:])).reshape(x.shape)
        L.append(model(x, xfft)[0].squeeze().item()); Y.append(int(y))
    return np.array(L), np.array(Y)

def fit_temperature(logits, labels, lr=0.01, iters=500):
    T = torch.nn.Parameter(torch.ones(1, dtype=torch.float32, device=DEVICE))
    optim = torch.optim.LBFGS([T], max_iter=iters, line_search_fn="strong_wolfe")
    y = torch.tensor(labels, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    L = torch.tensor(logits, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    bce = torch.nn.BCEWithLogitsLoss()
    def closure():
        optim.zero_grad()
        loss = bce(L / T.clamp(min=1e-3), y)
        loss.backward()
        return loss
    optim.step(closure)
    return float(T.detach().cpu().item())

def reliability_diagram(probs, labels, bins=10, path="figures/reliability_val.pdf"):
    import numpy as np
    bins_edges = np.linspace(0,1,bins+1)
    ece = 0.0
    plt.figure(figsize=(5,4))
    for i in range(bins):
        lo, hi = bins_edges[i], bins_edges[i+1]
        mask = (probs>=lo) & (probs<hi)
        if mask.sum()==0: continue
        conf = probs[mask].mean(); acc = labels[mask].mean()
        ece += (mask.sum()/len(probs)) * abs(acc-conf)
        plt.bar((lo+hi)/2, acc, width=(hi-lo)*0.9, alpha=0.7)
    plt.plot([0,1],[0,1],'--',lw=1)
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(f"Reliability (ECE≈{ece:.3f})")
    Path("figures").mkdir(exist_ok=True)
    plt.tight_layout(); plt.savefig(path)
    return ece

if __name__ == "__main__":
    ds_val = VideoFaceDataset(ROOT, "val", T=T, transform=val_tf)
    model = MMHTC().to(DEVICE); model.load_state_dict(torch.load("mmhtc_best.pth", map_location=DEVICE))

    logits, labels = logits_labels(model, ds_val)
    # uncalibrated
    probs = 1/(1+np.exp(-logits))
    ece0 = reliability_diagram(probs, labels, path="figures/reliability_val_uncal.pdf")
    # calibrated
    T = fit_temperature(logits, labels)
    probs_T = 1/(1+np.exp(-(logits / T)))
    eceT = reliability_diagram(probs_T, labels, path="figures/reliability_val_calibrated.pdf")
    print(f"Temp={T:.3f} | ECE (raw)={ece0:.4f} → (calibrated)={eceT:.4f}")
