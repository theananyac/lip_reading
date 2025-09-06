import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
from process_miracle_dataset import extract_lip  # Ensure this exists
import time

# Load model and classes
model = load_model("utils/lip_model.h5")
classes = np.load("utils/classes.npy")

IMG_SIZE = (64, 64)
FRAME_COUNT = 10
MOVEMENT_THRESHOLD = 5  # Adjust this as needed
DISPLAY_DURATION = 3  # seconds

frame_buffer = deque(maxlen=FRAME_COUNT)
prev_frame = None
predicted_word = ""
last_pred_time = 0

# Webcam
cap = cv2.VideoCapture(0)
print("📸 Live webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    mouth = extract_lip(frame)

    if mouth is not None:
        mouth_gray = cv2.cvtColor(cv2.resize(mouth, IMG_SIZE), cv2.COLOR_BGR2GRAY)

        # Calculate lip movement
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, mouth_gray)
            movement = np.mean(diff)
        else:
            movement = 0.0

        prev_frame = mouth_gray.copy()

        cv2.putText(frame, f"Movement: {movement:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if movement > MOVEMENT_THRESHOLD:
            mouth_gray = mouth_gray / 255.0
            frame_buffer.append(mouth_gray)

        if len(frame_buffer) == FRAME_COUNT:
            input_data = np.array(frame_buffer)[np.newaxis, ..., np.newaxis]
            prediction = model.predict(input_data, verbose=0)
            predicted_word = classes[np.argmax(prediction)]
            last_pred_time = time.time()
            print("✅ Predicted:", predicted_word)
            frame_buffer.clear()

    # Clear word after DISPLAY_DURATION
    if time.time() - last_pred_time < DISPLAY_DURATION:
        cv2.putText(frame, f"Predicted: {predicted_word}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Lip Reading - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
