import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_DIR = "/Users/dhruvshrinet/Desktop/Thesis/"
TEST_DIR = "/Users/dhruvshrinet/Desktop/Thesis/output_path"  # <---- Change this to your test dataset path
BATCH_SIZE = 32
IMAGE_SIZE = 224

# Data transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model loaders
def load_cnn_model():
    from torchvision import models
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def load_vit_model():
    import timm
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1)
    return model

def load_swin_model():
    import timm
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=1)
    return model

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds.cpu() == labels.cpu().int()).sum().item()
            total += labels.size(0)
    return 100 * correct / total

if __name__ == "__main__":
    print(f"Using device: {DEVICE}\n")

    models_to_eval = {
        "CNN (ResNet50)": ("resnet50_deepfake_model.pth", load_cnn_model),
        "ViT (Vision Transformer)": ("vit_model.pth", load_vit_model),
        "Swin Transformer": ("swin_deepfake_model.pth", load_swin_model),
    }

    for name, (filename, loader_fn) in models_to_eval.items():
        model_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(model_path):
            print(f"⚠️ Skipping {name}: file not found at {model_path}")
            continue

        model = loader_fn().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        acc = evaluate(model, test_loader)
        print(f"{name} Accuracy: {acc:.2f}%")