import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import classification_report
from tqdm import tqdm
import timm  # PyTorch image models

# Paths
TRAIN_DIR = "/Users/dhruvshrinet/Desktop/Thesis/output_path"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset and loader
dataset = ImageFolder(TRAIN_DIR, transform=transform)

# Train/val split (80/20)
from torch.utils.data import random_split
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Load ViT model
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1)
model.to(DEVICE)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
preds, targets = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds += (torch.sigmoid(outputs) > 0.5).cpu().int().squeeze().tolist()
        targets += labels.tolist()

print("\nViT Evaluation Results:\n")
print(classification_report(targets, preds, target_names=dataset.classes))