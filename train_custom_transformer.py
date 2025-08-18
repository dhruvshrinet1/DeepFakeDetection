import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --------- Model Definition ---------
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.activations = None  # For Grad-CAM hook

    def forward(self, x):
        x = self.proj(x)  # B x embed_dim x H/P x W/P
        self.activations = x  # Save conv feature map for Grad-CAM
        x = x.flatten(2)  # B x embed_dim x N_patches
        x = x.transpose(1, 2)  # B x N_patches x embed_dim
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=2, mlp_ratio=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*mlp_ratio)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.transpose(0, 1)  # N_patches x B x embed_dim
        x = self.transformer(x)
        x = x.transpose(0, 1)  # B x N_patches x embed_dim
        return x

class ST_HTC(nn.Module):
    def __init__(self, num_classes=1, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.spatial_transformer = SpatialTransformer(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.spatial_transformer(x)
        x = x.mean(dim=1)  # Global average pooling over patches
        x = self.classifier(x)
        return x

# --------- Config ---------
DATA_DIR = "/Users/dhruvshrinet/Desktop/Thesis/output_path"  # CHANGE THIS
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "st_htc_deepfake_model.pth"

# --------- Data ---------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --------- Training & Validation ---------
def train_one_epoch(model, optimizer, criterion, loader):
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            pred_labels = (torch.sigmoid(outputs) > 0.5).cpu().int().squeeze()
            preds += pred_labels.tolist()
            targets += labels.tolist()
    return preds, targets

# --------- Grad-CAM ---------
def generate_gradcam(model, input_tensor, class_idx=0):
    model.eval()
    input_tensor = input_tensor.to(DEVICE)

    # Forward pass
    output = model(input_tensor.unsqueeze(0))
    pred_prob = torch.sigmoid(output)[0][0].item()

    # Zero grads
    model.zero_grad()
    target = output[0, 0]
    target.backward()

    # Get activations and gradients from patch_embed conv layer
    activations = model.patch_embed.activations  # B x C x H x W
    gradients = model.patch_embed.proj.weight.grad if model.patch_embed.proj.weight.grad is not None else None
    
    # For Grad-CAM, we want gradients of output w.r.t activations
    # Hooking gradients directly not trivial here — instead we use hooks below
    
    # We’ll do a simplified Grad-CAM implementation:
    # Register hooks to capture gradients
    gradients = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle = model.patch_embed.proj.register_backward_hook(backward_hook)

    # Forward + backward again to get gradients
    output = model(input_tensor.unsqueeze(0))
    model.zero_grad()
    target = output[0, 0]
    target.backward()

    handle.remove()

    grads = gradients[0].cpu().data.numpy()[0]  # C x H x W
    acts = model.patch_embed.activations.cpu().data.numpy()[0]  # C x H x W

    weights = np.mean(grads, axis=(1, 2))  # C

    cam = np.zeros(acts.shape[1:], dtype=np.float32)  # H x W

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)  # Normalize 0-1

    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    return cam, pred_prob

def show_gradcam(img_tensor, cam, title=None):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5) + 0.5  # unnormalize
    img = np.clip(img, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1] / 255.0  # BGR to RGB

    overlayed = heatmap * 0.4 + img * 0.6
    overlayed = np.clip(overlayed, 0, 1)

    plt.figure(figsize=(6,6))
    plt.imshow(overlayed)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# --------- Main ---------
def main():
    model = ST_HTC().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader)
        preds, targets = validate(model, val_loader)
        
        val_acc = accuracy_score(targets, preds)
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(classification_report(targets, preds, target_names=full_dataset.classes))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"📦 Saved best model at epoch {epoch+1}")

    # Grad-CAM demo on some validation samples
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    print("\nGenerating Grad-CAM visualizations on val samples...")

    samples_shown = 0
    for imgs, labels in val_loader:
        for i in range(len(imgs)):
            cam, prob = generate_gradcam(model, imgs[i])
            label_str = full_dataset.classes[labels[i]]
            title = f"Pred Prob: {prob:.2f} | True: {label_str}"
            show_gradcam(imgs[i], cam, title)
            samples_shown += 1
            if samples_shown >= 5:
                return

if __name__ == "__main__":
    main()