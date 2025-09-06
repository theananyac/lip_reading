import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# Mapping of word index to actual word
word_map = {
    "01": "bin", "02": "blue", "03": "green", "04": "please", "05": "soon",
    "06": "red", "07": "white", "08": "yellow", "09": "stop", "10": "go"
}

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Lip landmark indices
UPPER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIPS = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Dataset paths
root_dir = r"C:\Users\spoor\OneDrive\Desktop\lip_mini\dataset"
output_dir = "mouth_dataset"
os.makedirs(output_dir, exist_ok=True)

def extract_lip(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None

    h, w, _ = frame.shape
    lip_points = []
    for idx in UPPER_LIPS + LOWER_LIPS:
        pt = results.multi_face_landmarks[0].landmark[idx]
        lip_points.append((int(pt.x * w), int(pt.y * h)))

    x_coords = [pt[0] for pt in lip_points]
    y_coords = [pt[1] for pt in lip_points]
    x_min, x_max = max(min(x_coords)-10, 0), min(max(x_coords)+10, w)
    y_min, y_max = max(min(y_coords)-10, 0), min(max(y_coords)+10, h)
    return frame[y_min:y_max, x_min:x_max]

def process():
    for subject in os.listdir(root_dir):
        word_path = os.path.join(root_dir, subject, "words")
        if not os.path.isdir(word_path):
            continue
        for word_id in os.listdir(word_path):
            word = word_map.get(word_id)
            if word is None:
                continue
            sample_path = os.path.join(word_path, word_id)
            for sample in tqdm(os.listdir(sample_path), desc=f"{subject}/{word}"):
                sample_dir = os.path.join(sample_path, sample)
                out_dir = os.path.join(output_dir, word, f"{subject}_{sample}")
                os.makedirs(out_dir, exist_ok=True)
                frames = sorted([f for f in os.listdir(sample_dir) if f.startswith("color")])
                for idx, frame_name in enumerate(frames):
                    img_path = os.path.join(sample_dir, frame_name)
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                    mouth = extract_lip(frame)
                    if mouth is None:
                        continue
                    resized = cv2.resize(mouth, (100, 50))
                    cv2.imwrite(os.path.join(out_dir, f"frame{idx}.jpg"), resized)

if __name__ == "__main__":
    process()
    print("✅ Finished processing MIRACL-VC1 dataset.")
