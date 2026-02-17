import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import os

print("🚀 Starting Training Process...")

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

print(f" Total training samples: {len(sentences)}")

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
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(labels_encoded)), activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(" Training Model...")
model.fit(padded_sequences, labels_encoded, epochs=200, verbose=1)

# ===============================
# Create Model Folder
# ===============================
os.makedirs("model", exist_ok=True)

# ===============================
# Save Model (Modern Format)
# ===============================
model_path = os.path.join("model", "model.keras")
model.save(model_path)

# Save Tokenizer
tokenizer_path = os.path.join("model", "tokenizer.pkl")
pickle.dump(tokenizer, open(tokenizer_path, 'wb'))

# Save Label Encoder
encoder_path = os.path.join("model", "label_encoder.pkl")
pickle.dump(label_encoder, open(encoder_path, 'wb'))

print("Training Complete!")
print(" Model saved at:", model_path)
print(" Tokenizer saved at:", tokenizer_path)
print(" Label encoder saved at:", encoder_path)
