import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

with open('intents.json') as file:
    data = json.load(file)

sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenize sentences
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

max_len = 20
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Build LSTM model
model = Sequential()
model.add(Embedding(2000, 128, input_length=max_len))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(labels)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(padded_sequences, labels, epochs=200)

# Save everything
model.save('model.h5')
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
pickle.dump(label_encoder, open('label_encoder.pkl', 'wb'))

print("LSTM Model trained successfully!")
