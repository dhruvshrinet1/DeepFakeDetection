import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import timm

# Config
DATA_DIR = "/Users/dhruvshrinet/Desktop/Thesis/output_path"
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset & Split (90% train, 10% val)
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = full_dataset.classes
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Swin Transformer model (binary classification)
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=1)
model = model.to(DEVICE)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
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
        preds += (torch.sigmoid(outputs) > 0.5).int().cpu().squeeze().tolist()
        targets += labels.tolist()

# Classification report
print(classification_report(targets, preds, target_names=class_names))

# Save model
torch.save(model.state_dict(), "swin_deepfake_model.pth")
print("✅ Model saved as swin_deepfake_model.pth")