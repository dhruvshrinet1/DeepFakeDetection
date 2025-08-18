import os, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np

# --------------------- Reproducibility ---------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True  # speed
torch.backends.cudnn.deterministic = False

# --------------------- Custom JPEG transform ---------------------
class RandomJPEG(object):
    def __init__(self, qmin=70, qmax=100, p=0.5):
        self.qmin, self.qmax, self.p = qmin, qmax, p
    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        q = random.randint(self.qmin, self.qmax)
        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

# --------------------- Model (GAP head; tiny but solid) ---------------------
class CustomDeepFakeCNN(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        # 224 -> 112 -> 56 -> 28 -> 14
        self.block1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
                                    nn.Conv2d(base, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(base, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, padding=1), nn.BatchNorm2d(base*4), nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.block4 = nn.Sequential(nn.Conv2d(base*4, base*8, 3, padding=1), nn.BatchNorm2d(base*8), nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Dropout(0.3),
                                  nn.Linear(base*8, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 1))
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        x = self.gap(x)
        return self.head(x)

# --------------------- Config ---------------------
DATA_DIR   = "/Users/dhruvshrinet/Desktop/Thesis/output_path"  # <-- set me
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS     = 20
LR_MAX     = 3e-4
WEIGHT_DEC = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH  = "custom_cnn_deepfake_model.pth"

# --------------------- Transforms ---------------------
train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    RandomJPEG(70, 100, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])
val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# --------------------- Dataset + stratified split ---------------------
full_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)  # temp; we’ll reset transform for val subset
labels = [y for _, y in full_ds.samples]
indices = np.arange(len(labels))
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=SEED)
train_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=train_tf), train_idx)
val_ds   = Subset(datasets.ImageFolder(DATA_DIR, transform=val_tf),   val_idx)

# class weights for imbalance
train_labels = np.array(labels)[train_idx]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# pos_weight for BCE if classes imbalanced
pos = class_counts[1] if len(class_counts) > 1 else 1
neg = class_counts[0] if len(class_counts) > 0 else 1
pos_weight = torch.tensor([neg / max(pos,1)], device=DEVICE, dtype=torch.float32)

# --------------------- Training utilities ---------------------
def cosine_lr(step, total_steps, lr_max, lr_min=1e-6, warmup=500):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    return lr_min + 0.5*(lr_max - lr_min)*(1 + math.cos(math.pi * t))

def train_one_epoch(model, optimizer, scaler, criterion, loader, total_steps, step0):
    model.train()
    running = 0.0; step = step0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        # cosine LR
        lr = cosine_lr(step, total_steps, LR_MAX)
        for g in optimizer.param_groups: g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        running += loss.item(); step += 1
    return running / max(1, len(loader)), step

@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_probs = []; all_preds = []; all_targets = []
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        preds = (probs > 0.5).astype(np.int32)
        all_probs.append(probs); all_preds.append(preds)
        all_targets.append(labels.numpy())
    probs  = np.concatenate(all_probs)
    preds  = np.concatenate(all_preds)
    target = np.concatenate(all_targets)
    acc = accuracy_score(target, preds)
    try:
        auc = roc_auc_score(target, probs)
    except:
        auc = float('nan')
    return acc, auc, preds, target

# --------------------- Main ---------------------
def main():
    model = CustomDeepFakeCNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DEC)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    total_steps = EPOCHS * len(train_loader)
    step = 0
    best = -1.0

    for epoch in range(1, EPOCHS+1):
        train_loss, step = train_one_epoch(model, optimizer, scaler, criterion, train_loader, total_steps, step)
        val_acc, val_auc, preds, targets = validate(model, val_loader)

        print(f"\nEpoch {epoch}/{EPOCHS} | Train Loss {train_loss:.4f} | Val Acc {val_acc:.4f} | Val AUC {val_auc:.4f}")
        # Show per-class report (0=fake/real depending on folder order)
        print(classification_report(targets, preds, target_names=datasets.ImageFolder(DATA_DIR).classes))

        score = val_auc if not np.isnan(val_auc) else val_acc
        if score > best:
            best = score
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"📦 Saved best model to {SAVE_PATH} (score={best:.4f})")

if __name__ == "__main__":
    main()
