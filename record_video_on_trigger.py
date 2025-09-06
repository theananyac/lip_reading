import os
import cv2
import time
import numpy as np
import mediapipe as mp
import sounddevice as sd
import scipy.io.wavfile as wav
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model and classes
model = load_model("utils/lip_model.h5")
class_names = np.load("utils/classes.npy")

# Configuration
frame_count = 10
img_size = (64, 64)
trigger_word = "go"

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
UPPER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIPS = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

def extract_lip(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)
    if not result.multi_face_landmarks:
        return None
    h, w, _ = frame.shape
    points = [(int(landmark.x * w), int(landmark.y * h)) for idx in UPPER_LIPS + LOWER_LIPS
              for landmark in [result.multi_face_landmarks[0].landmark[idx]]]
    x_coords = [pt[0] for pt in points]
    y_coords = [pt[1] for pt in points]
    x1, x2 = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
    y1, y2 = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)
    return frame[y1:y2, x1:x2]

def predict_word(frames):
    frames = np.array(frames) / 255.0
    frames = np.expand_dims(frames, axis=-1)
    frames = np.expand_dims(frames, axis=0)
    prediction = model.predict(frames)
    return class_names[np.argmax(prediction)]

def record_audio(filename="audio.wav", duration=10, samplerate=16000):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, samplerate, audio)

def record_video(duration=10, filename="video.avi"):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    start = time.time()
    while int(time.time() - start) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow("Recording Window", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    out.release()
    cv2.destroyWindow("Recording Window")

def merge_and_denoise():
    print("\n🎵 Merging audio and video...")
    os.system("ffmpeg -y -i video.avi -i audio.wav -c:v copy -c:a aac output.mp4")
    print("🎵 Extracting audio...")
    os.system("ffmpeg -y -i output.mp4 -vn -acodec pcm_s16le extracted_audio.wav")

    import noisereduce as nr
    rate, data = wav.read("extracted_audio.wav")
    reduced = nr.reduce_noise(y=data.astype(float), sr=rate)
    wav.write("cleaned_output.wav", rate, reduced.astype(np.int16))
    print("✅ Denoised audio saved as 'cleaned_output.wav'")

# Main trigger loop
cap = cv2.VideoCapture(0)
frame_buffer = []

print("👀 Waiting for trigger word...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    mouth = extract_lip(frame)
    if mouth is not None:
        mouth_resized = cv2.resize(mouth, img_size)
        mouth_gray = cv2.cvtColor(mouth_resized, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(mouth_gray)

    if len(frame_buffer) == frame_count:
        word = predict_word(frame_buffer)
        print("🧠 Predicted:", word)
        frame_buffer = []

        if word == trigger_word:
            print("\n🚀 Trigger word detected! Starting recording...")
            video_thread = threading.Thread(target=record_video, args=(10,))
            audio_thread = threading.Thread(target=record_audio, args=("audio.wav", 10))
            video_thread.start()
            audio_thread.start()
            video_thread.join()
            audio_thread.join()
            merge_and_denoise()
            break

    cv2.imshow("Prediction Window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
