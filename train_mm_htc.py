# train_mm_htc.py
import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
from dataset_video import VideoFaceDataset
from fft_feats import rgb_to_fft_amp
from model_mm_htc import MMHTC
from torchvision import transforms
import argparse
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE=224; BATCH=8; T=8; EPOCHS=10; LR=3e-4; WD=1e-4

train_tf = transforms.Compose([
  transforms.Resize((IMG_SIZE,IMG_SIZE)),
  transforms.RandomHorizontalFlip(0.5),
  transforms.ColorJitter(0.2,0.2,0.2,0.05),
  transforms.ToTensor(),
  transforms.Normalize([0.5]*3,[0.5]*3)
])
val_tf = transforms.Compose([
  transforms.Resize((IMG_SIZE,IMG_SIZE)),
  transforms.ToTensor(),
  transforms.Normalize([0.5]*3,[0.5]*3)
])

def make_loader(root, split, tf, shuffle):
    ds = VideoFaceDataset(root, split, T=T, transform=tf)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=4, pin_memory=True, drop_last=False)

def train_epoch(model, opt, scaler, crit, loader, aux_w=0.2):
    model.train(); losses=[]
    for x, y in tqdm(loader, leave=False):
        x = x.to(DEVICE); y = y.float().unsqueeze(1).to(DEVICE)
# before:
# x_fft = rgb_to_fft_amp(x.reshape(-1, *x.shape[2:])).reshape(x.shape)

# after:
        if use_fft:
            x_fft = rgb_to_fft_amp(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
        else:
            x_fft = None
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):
            logit, aux = model(x, x_fft)
            loss = crit(logit, y)
            # weak frame labels = video label
            loss = loss + aux_w * nn.BCEWithLogitsLoss()(aux, y.repeat(1, aux.shape[1]))
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); probs=[]; tgts=[]
    for x, y in tqdm(loader, leave=False):
        x = x.to(DEVICE); y = y.numpy()
        if use_fft:
            x_fft = rgb_to_fft_amp(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
        else:
            x_fft = None
        
        p = torch.sigmoid(model(x, x_fft)[0]).squeeze(1).cpu().numpy()
        probs.append(p); tgts.append(y)
    probs = np.concatenate(probs); tgts = np.concatenate(tgts)
    preds = (probs>0.5).astype(int)
    return accuracy_score(tgts,preds), roc_auc_score(tgts,probs), f1_score(tgts,preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rgb_fft_temporal",
                        choices=["rgb_only","fft_only","rgb_fft","rgb_fft_temporal"],
                        help="Which ablation/baseline to run.")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    EPOCHS = args.epochs  # override epochs from CLI
    mode = args.mode
    use_rgb = mode in ["rgb_only","rgb_fft","rgb_fft_temporal"]
    use_fft = mode in ["fft_only","rgb_fft","rgb_fft_temporal"]
    use_temporal = mode in ["rgb_fft_temporal"]
    print(f"[Config] mode={mode} | use_rgb={use_rgb} use_fft={use_fft} use_temporal={use_temporal}")
    ROOT = "/Users/dhruvshrinet/Downloads/deepfake-detection-challenge/train_sample_videos/dataset_faces"
    train_loader = make_loader(ROOT, "train", train_tf, shuffle=True)
    val_loader   = make_loader(ROOT, "val",   val_tf,   shuffle=False)

    # imbalance handling
    # compute class counts from train_loader.dataset.samples if needed → pos_weight
    pos_weight = torch.tensor([1.0], device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = MMHTC(use_rgb=use_rgb, use_fft=use_fft, use_temporal=use_temporal).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda')
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_auc = -1
    for e in range(1, EPOCHS+1):
        tl = train_epoch(model, opt, scaler, crit, train_loader)
        acc, auc, f1 = evaluate(model, val_loader)
        print(f"Epoch {e}/{EPOCHS} | loss {tl:.4f} | val acc {acc:.4f} | val auc {auc:.4f} | val f1 {f1:.4f}")
        sched.step()
        if auc > best_auc:
            best_auc = auc
            ckpt_path = f"mmhtc_best_{mode}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"saved {ckpt_path}")
