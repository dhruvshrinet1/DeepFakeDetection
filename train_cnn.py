import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# Paths
DATA_DIR = "/Users/dhruvshrinet/Desktop/Thesis/output_path"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load dataset and split into train and val
full_dataset = ImageFolder(DATA_DIR, transform=transform)
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Model: ResNet50
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary output
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(model)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

# Validation
model.eval()
preds, targets = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        pred_labels = (torch.sigmoid(outputs) > 0.5).cpu().int().squeeze()
        preds += pred_labels.tolist()
        targets += labels.tolist()

print("\nValidation Results:")
print(classification_report(targets, preds, target_names=full_dataset.classes))