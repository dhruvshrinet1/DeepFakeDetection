# eval_mm_htc.py
import argparse, json, numpy as np, torch
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, roc_curve,
    balanced_accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
from torchvision import transforms

from dataset_video import VideoFaceDataset
from model_mm_htc import MMHTC

# ---- FFT import tolerant to file name ----
try:
    from fft_feats import rgb_to_fft_amp
except Exception:
    from fff_feats import rgb_to_fft_amp  # fallback if your file is named fff_feats.py

# ---------------- config ----------------
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
T        = 8

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

@torch.no_grad()
def evaluate_one(model, ds, use_fft: bool):
    model.eval()
    probs, tgts = [], []
    for x, y in ds:
        x = x.unsqueeze(0).to(DEVICE)  # [1,T,3,224,224]
        xfft = None
        if use_fft:
            xfft = rgb_to_fft_amp(x.reshape(-1, *x.shape[2:])).reshape(x.shape)
        p = torch.sigmoid(model(x, xfft)[0]).squeeze().item()
        probs.append(p); tgts.append(int(y))
    probs = np.array(probs); tgts = np.array(tgts)
    return probs, tgts

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint to evaluate")
    p.add_argument("--mode", type=str, default="rgb_fft_temporal",
                   choices=["rgb_only","fft_only","rgb_fft","rgb_fft_temporal"],
                   help="Must match how the checkpoint was trained.")
    p.add_argument("--root", type=str,
                   default="/Users/dhruvshrinet/Downloads/deepfake-detection-challenge/train_sample_videos/dataset_faces",
                   help="Dataset root with train/ and val/ splits")
    p.add_argument("--tag", type=str, default=None, help="Optional tag to suffix output filenames")
    args = p.parse_args()

    use_rgb      = args.mode in ["rgb_only","rgb_fft","rgb_fft_temporal"]
    use_fft      = args.mode in ["fft_only","rgb_fft","rgb_fft_temporal"]
    use_temporal = args.mode in ["rgb_fft_temporal"]
    print(f"[EVAL] mode={args.mode} | use_rgb={use_rgb} use_fft={use_fft} use_temporal={use_temporal}")
    print(f"[EVAL] ROOT: {args.root}")

    # dataset + quick sanity
    ds_val = VideoFaceDataset(args.root, "val", T=T, transform=val_tf)
    print(f"[EVAL] val samples: {len(ds_val)}")
    if len(ds_val) == 0:
        raise RuntimeError(f"No validation samples found under {args.root}/val")

    # build matching model & load weights
    model = MMHTC(use_rgb=use_rgb, use_fft=use_fft, use_temporal=use_temporal).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print(f"[warn] missing keys: {len(missing)} (expected for ablations)")
    if unexpected:print(f"[warn] unexpected keys: {len(unexpected)}")

    # evaluate
    probs, tgts = evaluate_one(model, ds_val, use_fft=use_fft)
    preds05 = (probs >= 0.5).astype(int)

    # metrics
    try:
        auc = roc_auc_score(tgts, probs)
    except Exception:
        auc = float("nan")
    acc05 = accuracy_score(tgts, preds05)
    f105  = f1_score(tgts, preds05, zero_division=0)

    # Youden J operating point (on val)
    fpr, tpr, thr = roc_curve(tgts, probs)
    j = tpr - fpr
    t_star = thr[np.argmax(j)]
    predsJ = (probs >= t_star).astype(int)
    balacc = balanced_accuracy_score(tgts, predsJ)
    tn, fp, fn, tp = confusion_matrix(tgts, predsJ).ravel()

    # outputs
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    suffix = f"_{args.mode}" if args.tag is None else f"_{args.tag}"
    # CSV
    import csv
    with open(out_dir/f"per_video_scores{suffix}.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["idx","label","prob","pred@0.5","pred@youden"])
        for i,(y,p,pr05,prJ) in enumerate(zip(tgts, probs, preds05, predsJ)):
            w.writerow([i,int(y),float(p),int(pr05),int(prJ)])
    # JSON
    with open(out_dir/f"results_summary{suffix}.json","w") as f:
        json.dump({
          "mode": args.mode,
          "ckpt": args.ckpt,
          "val_auc": float(auc),
          "val_acc@0.5": float(acc05),
          "val_f1@0.5": float(f105),
          "youden_t": float(t_star),
          "val_bal_acc@youden": float(balacc),
          "cm@youden": {"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
        }, f, indent=2)

    # ROC plot (only if AUC is defined)
    if not np.isnan(auc):
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'--',lw=1)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title(f"ROC (val) — {args.mode}")
        plt.tight_layout(); plt.savefig(out_dir/f"roc_val{suffix}.pdf")
        print(f"Saved ROC to {out_dir/f'roc_val{suffix}.pdf'}")

    print(f"VAL ({args.mode}): AUC={auc:.4f} | Acc@0.5={acc05:.4f} F1@0.5={f105:.4f} | "
          f"Youden t={t_star:.3f} BalAcc@Youden={balacc:.4f}")
