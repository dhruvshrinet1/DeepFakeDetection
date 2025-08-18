import os, json, cv2, torch, numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
from tqdm import tqdm
from PIL import Image

# -------------------- Config --------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 224
K_FRAMES = 16                  # exact frames per video
MIN_PROB = 0.90                # min face conf
MIN_BOX = 64                   # min face box size (pixels)
MARGIN = 20

VIDEO_DIR = '/Users/dhruvshrinet/Downloads/deepfake-detection-challenge/train_sample_videos'
OUTPUT_DIR = '/Users/dhruvshrinet/Downloads/deepfake-detection-challenge/train_sample_videos/output_path/new_output'
META_FILE = os.path.join(VIDEO_DIR, 'metadata.json')

# -------------------- Init ----------------------
torch.set_grad_enabled(False)
cv2.setNumThreads(0)

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=MARGIN,
    keep_all=False,            # choose single main face
    select_largest=True,
    post_process=True,
    device=DEVICE
)
mtcnn.eval()

# -------------------- Helpers -------------------
def pil_from_bgr(frame_bgr):
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

def uniform_indices(n, k):
    if n <= 0: return []
    k = min(k, n)
    return np.linspace(0, n-1, num=k, dtype=int).tolist()

def crop_with_box(frame_bgr, box, image_size=224, margin=20):
    # box: [x1,y1,x2,y2]
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:  # fallback if invalid box
        return cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return crop

def extract_faces_uniform(video_path, save_dir, k=16, min_prob=0.9, min_box=64):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(n) or n <= 0:
        # count manually (rare but happens)
        frames_cache = []
        while True:
            ret, f = cap.read()
            if not ret: break
            frames_cache.append(f)
        cap.release()
        indices = uniform_indices(len(frames_cache), k)
        frames = [frames_cache[i] for i in indices]
    else:
        indices = uniform_indices(n, k)
        frames = []
        idx_set = set(indices)
        i = 0
        while True:
            ret, f = cap.read()
            if not ret: break
            if i in idx_set:
                frames.append(f)
                if len(frames) == len(indices):
                    break
            i += 1
        cap.release()

    saved = 0
    last_good_box = None

    for j, frame_bgr in enumerate(frames):
        # Try detection with box/prob, so we can quality-filter
        pil_img = pil_from_bgr(frame_bgr)
        boxes, probs = mtcnn.detect(pil_img)
        face_img = None

        if boxes is not None and len(boxes) > 0:
            # pick best box
            best = int(np.argmax(probs))
            box, prob = boxes[best], float(probs[best] if probs is not None else 1.0)
            w = box[2] - box[0]; h = box[3] - box[1]
            if prob >= min_prob and w >= min_box and h >= min_box:
                face_img = crop_with_box(frame_bgr, box, IMAGE_SIZE, margin=MARGIN)
                last_good_box = box
        # Fallback: use last good box if current is weak
        if face_img is None and last_good_box is not None:
            face_img = crop_with_box(frame_bgr, last_good_box, IMAGE_SIZE, margin=MARGIN)
        # Final fallback: center-crop resized frame (so we ALWAYS save K frames)
        if face_img is None:
            resized = cv2.resize(frame_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            face_img = resized

        out_name = f"{Path(video_path).stem}_f{j:04d}.jpg"
        cv2.imwrite(str(Path(save_dir, out_name)), face_img)
        saved += 1

    return saved

# -------------------- Main loop ------------------
def main():
    with open(META_FILE, 'r') as f:
        metadata = json.load(f)

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    print(f"Found {len(video_files)} videos.")

    for vf in tqdm(video_files, desc="Extracting faces"):
        video_path = os.path.join(VIDEO_DIR, vf)
        if vf not in metadata:
            # DFDC has some stray files; safe to skip
            continue
        label = metadata[vf]['label'].upper()  # REAL / FAKE
        save_dir = os.path.join(OUTPUT_DIR, label, vf)  # keep video name folder (with .mp4)
        # skip if already extracted K frames
        if os.path.isdir(save_dir) and len([x for x in os.listdir(save_dir) if x.endswith('.jpg')]) >= K_FRAMES:
            continue
        try:
            saved = extract_faces_uniform(video_path, save_dir, k=K_FRAMES, min_prob=MIN_PROB, min_box=MIN_BOX)
        except Exception as e:
            print(f"[WARN] Failed {vf}: {e}")
            continue

if __name__ == "__main__":
    main()
