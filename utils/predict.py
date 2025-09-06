import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from process_miracle_dataset import extract_lip

# Load the trained model and class labels
model = load_model("utils/lip_model.h5")
classes = np.load("utils/classes.npy")

# Parameters
IMG_SIZE = (64, 64)
FRAME_COUNT = 10
VIDEO_PATH = "utils/test2_video.mp4"

# Capture video and extract lip regions
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    mouth = extract_lip(frame)
    if mouth is None:
        continue
    mouth = cv2.resize(mouth, IMG_SIZE)
    gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0
    frames.append(gray)
cap.release()

# Select exactly FRAME_COUNT frames
if len(frames) < FRAME_COUNT:
    raise Exception(f"Need at least {FRAME_COUNT} valid mouth frames.")
frames = frames[:FRAME_COUNT]

# Prepare input for prediction
X = np.array(frames).reshape(1, FRAME_COUNT, IMG_SIZE[0], IMG_SIZE[1], 1)
pred = model.predict(X)
predicted_label = classes[np.argmax(pred)]
print(f"✅ Predicted Word: {predicted_label}")
