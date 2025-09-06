import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, TimeDistributed, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Set dataset path
dataset_path = "mouth_dataset"
img_size = (64, 64)
frame_count = 10  # fixed number of frames per sequence

X = []
y = []
classes = sorted(os.listdir(dataset_path))
print(f"Detected classes: {classes}")

for label in classes:
    word_path = os.path.join(dataset_path, label)
    for sequence in os.listdir(word_path):
        seq_path = os.path.join(word_path, sequence)
        frames = sorted(os.listdir(seq_path))

        if len(frames) < frame_count:
            continue  # skip sequences that are too short

        frames = frames[:frame_count]
        frame_data = []
        for frame in frames:
            img_path = os.path.join(seq_path, frame)
            img = Image.open(img_path).convert("L")
            img = img.resize(img_size)
            img = np.array(img) / 255.0  # normalize pixel values
            frame_data.append(img)

        X.append(frame_data)
        y.append(label)

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y_encoded)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Add channel dimension to each image
X = X[..., np.newaxis]

# One-hot encode the labels
y = to_categorical(y, num_classes=len(classes))

# Split dataset
if len(np.unique(y.argmax(axis=1))) < 2:
    raise Exception("Need at least 2 classes with multiple samples for training.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y.argmax(axis=1), random_state=42
)

# Build the model
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(frame_count, 64, 64, 1)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Save model
model.save("utils/lip_model.h5")
print("✅ Model trained and saved as 'utils/lip_model.h5'")

# Save class labels
np.save("utils/classes.npy", label_encoder.classes_)
print("✅ Classes saved to 'utils/classes.npy'")
