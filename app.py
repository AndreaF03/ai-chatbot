from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json
import random
import pickle
import sqlite3
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Initialize Database
def init_db():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_message TEXT,
                  bot_response TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Load Model and NLP Files
model = tf.keras.models.load_model("model/model.h5")
tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('model/label_encoder.pkl', 'rb'))


with open("intents.json") as file:
    intents = json.load(file)

max_len = 20

# Predict Intent
def predict_class(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_len)
    result = model.predict(padded)[0]

    confidence = np.max(result)
    if confidence < 0.6:
        return None

    return label_encoder.inverse_transform([np.argmax(result)])[0]

# Get Bot Response
def get_response(tag):
    if tag:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/admin")
def admin():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM chats")
    total_messages = c.fetchone()[0]

    c.execute("SELECT * FROM chats ORDER BY id DESC LIMIT 10")
    recent_chats = c.fetchall()

    conn.close()

    return render_template("admin.html",
                           total_messages=total_messages,
                           recent_chats=recent_chats)

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.json["message"]

    tag = predict_class(msg)
    response = get_response(tag)

    # Save to database
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_message, bot_response, timestamp) VALUES (?, ?, ?)",
              (msg, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return jsonify({"response": response})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


