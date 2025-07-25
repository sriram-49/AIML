import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# === Paths ===
FEATURES_PATH = r"D:\Java prg\project\AIML\Project(AIML)\features"
MODEL_SAVE_PATH = r"D:\Java prg\project\AIML\Project(AIML)\models\genre_model.h5"

# === Load and prepare MFCC feature data ===
def load_data(max_len=130):  # Fixed number of time steps
    X, y = [], []
    for file in os.listdir(FEATURES_PATH):
        data_path = os.path.join(FEATURES_PATH, file)
        if not file.endswith('.npy'):
            continue

        data = np.load(data_path, allow_pickle=True).item()
        mfcc = data['mfcc']  # shape = (time_steps, 13)

        # Pad or truncate each MFCC to fixed length
        if mfcc.shape[0] < max_len:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_len]

        X.append(mfcc)
        y.append(data['label'])

    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], np.array(y)  # Add channel dimension for CNN

# === Load data ===
X, y = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Build CNN model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D((3, 3)),                     # OK after first conv layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((3, 1)),                     # ✅ FIXED: width is now safe
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')           # Update to match your genre count
])


# === Compile and train ===
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

# === Save trained model ===
model.save(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")
