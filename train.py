import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
import datetime

print("🚀 Starting Training Process...")

# Create model folder
os.makedirs("model", exist_ok=True)

# ===============================
# Load Dataset
# ===============================
with open('intents.json') as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

print(f"Total training samples: {len(sentences)}")

# ===============================
# Encode Labels
# ===============================
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# ===============================
# Tokenize Sentences
# ===============================
vocab_size = 3000
max_len = 20

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# ===============================
# Build LSTM Model
# ===============================
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(labels_encoded)), activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint_path = os.path.join("model", "best_model.h5")

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("Training Model...")
history = model.fit(
    padded_sequences,
    labels_encoded,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ===============================
# Save Metadata
# ===============================
final_accuracy = float(max(history.history['accuracy']))
val_accuracy = float(max(history.history['val_accuracy']))

metadata = {
    "timestamp": str(datetime.datetime.now()),
    "samples": len(sentences),
    "vocab_size": vocab_size,
    "max_len": max_len,
    "train_accuracy": final_accuracy,
    "validation_accuracy": val_accuracy
}

metadata_path = os.path.join("model", "training_metadata.json")

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("Training metadata saved at:", metadata_path)

# ===============================
# Save Model (.h5 format)
# ===============================
version_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = os.path.join("model", "model.h5")
versioned_model_path = os.path.join("model", f"model_{version_time}.h5")

model.save(versioned_model_path)
model.save(model_path)

print("Versioned model saved at:", versioned_model_path)
print("Production model saved at:", model_path)

# ===============================
# Save Tokenizer
# ===============================
tokenizer_path = os.path.join("model", "tokenizer.pkl")
pickle.dump(tokenizer, open(tokenizer_path, 'wb'))

# ===============================
# Save Label Encoder
# ===============================
encoder_path = os.path.join("model", "label_encoder.pkl")
pickle.dump(label_encoder, open(encoder_path, 'wb'))

print("Tokenizer saved at:", tokenizer_path)
print("Label encoder saved at:", encoder_path)

print("🎉 Training Complete!")
