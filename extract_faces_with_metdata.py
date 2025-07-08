import os
import cv2
import torch
import json
from facenet_pytorch import MTCNN
from tqdm import tqdm

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN detector
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=device)

# Paths
VIDEO_DIR = '/Users/dhruvshrinet/Downloads/deepfake-detection-challenge/train_sample_videos'
OUTPUT_DIR = '/Users/dhruvshrinet/Downloads/deepfake-detection-challenge/train_sample_videos/output_path'
METADATA_FILE = os.path.join(VIDEO_DIR, 'metadata.json')

# Load metadata
with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

# Function to extract and save face images
def extract_faces(video_path, save_dir, every_n_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n_frames == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(img)
            if face is not None:
                face = face.permute(1, 2, 0).byte().cpu().numpy()
    
    # Skip saving if the image is mostly black
            if face.mean() < 5:
                print(f"Skipped dark frame at {video_name}_{frame_idx}")
                continue
    
            filename = f"{video_name}_{frame_idx}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                    
            saved += 1
            frame_idx += 1
            cap.release()
    return saved


# Loop through videos and extract faces
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]

print(video_files)
for vf in tqdm(video_files, desc="Extracting faces"):
    video_path = os.path.join(VIDEO_DIR, vf)
    video_name = os.path.basename(video_path)  # Keep .mp4 extension
    # Skip if not found in metadata
    if video_name not in metadata:
        print(f"Skipping {video_name}: not in metadata.")
        continue

    label = metadata[video_name]['label'].upper()  # 'FAKE' or 'REAL'
    save_path = os.path.join(OUTPUT_DIR, label, video_name)
    os.makedirs(save_path, exist_ok=True)
    extract_faces(video_path, save_path)