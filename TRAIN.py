import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import librosa

# ── CONFIG ──
RECORDINGS_DIR = "recordings"
MODEL_SAVE_PATH = "models/best_model.h5"
SR = 8000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128
IMG_SIZE = (64, 64)

def extract_mel(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
        if y is None or len(y) == 0:
            return None
        # Normalize audio
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.max() == mel_db.min():
            return None
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        if mel_db.shape[1] < IMG_SIZE[1]:
            mel_db = np.pad(mel_db, ((0,0),(0, IMG_SIZE[1]-mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :IMG_SIZE[1]]
        if mel_db.shape != IMG_SIZE:
            return None
        return mel_db[..., np.newaxis]
    except Exception as e:
        print(f"Error: {file_path} — {e}")
        return None

def augment_audio(y, sr):
    """Create augmented versions of audio."""
    augmented = []
    # Add noise
    noise = np.random.randn(len(y)) * 0.005
    augmented.append(y + noise)
    # Time shift
    shift = int(sr * 0.1)
    augmented.append(np.roll(y, shift))
    # Pitch shift
    try:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1))
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1))
    except:
        pass
    # Speed change
    try:
        augmented.append(librosa.effects.time_stretch(y, rate=0.9))
        augmented.append(librosa.effects.time_stretch(y, rate=1.1))
    except:
        pass
    return augmented

def extract_mel_from_audio(y, sr):
    try:
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.max() == mel_db.min():
            return None
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        if mel_db.shape[1] < IMG_SIZE[1]:
            mel_db = np.pad(mel_db, ((0,0),(0, IMG_SIZE[1]-mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :IMG_SIZE[1]]
        if mel_db.shape != IMG_SIZE:
            return None
        return mel_db[..., np.newaxis]
    except:
        return None

# ── LOAD DATA ──
print("Loading recordings...")
X, y = [], []

files = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".wav")]
total = len(files)

for i, fname in enumerate(files):
    if i % 100 == 0:
        print(f"  Processing {i}/{total}...")
    try:
        digit = int(fname.split("_")[0])
        path = os.path.join(RECORDINGS_DIR, fname)
        feat = extract_mel(path)
        if feat is not None:
            X.append(feat)
            y.append(digit)
            # Augment each sample
            audio, sr = librosa.load(path, sr=SR)
            for aug_audio in augment_audio(audio, sr):
                aug_feat = extract_mel_from_audio(aug_audio, sr)
                if aug_feat is not None:
                    X.append(aug_feat)
                    y.append(digit)
    except:
        continue

X = np.array(X)
y = np.array(y)
print(f"\n✅ Loaded {len(X)} samples (with augmentation) across {len(set(y))} digits")
for d in sorted(set(y)):
    print(f"   Digit {d}: {sum(y==d)} samples")

# ── SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── IMPROVED MODEL ──
model = tf.keras.Sequential([
    # Block 1
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(64,64,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),

    # Block 2
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),

    # Block 3
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.4),

    # Block 4
    tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    # Dense
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── TRAIN ──
print("\nTraining...")
history = model.fit(
    X_train, y_train,
    epochs=80,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=6,
            factor=0.5,
            min_lr=0.00001,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
)

# ── EVALUATE ──
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Final Test Accuracy: {acc*100:.2f}%")

# ── SAVE ──
os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
print("\nNext step: upload models/best_model.h5 to GitHub!")