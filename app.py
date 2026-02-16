from flask import Flask, render_template, request, jsonify, redirect, session
import tensorflow as tf
import numpy as np
import json
import random
import pickle
import sqlite3
import os
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import check_password_hash, generate_password_hash
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import subprocess

# ==============================
# Initialize Flask App
# ==============================
app = Flask(__name__)
app.secret_key = "supersecretkey"

# ==============================
# Login Manager Setup
# ==============================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# ==============================
# Initialize Database
# ==============================
def init_db():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_message TEXT,
                  bot_response TEXT,
                  emotion TEXT,
                  confidence REAL,
                  timestamp TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS unknown_queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_message TEXT,
                  timestamp TEXT,
                  resolved INTEGER DEFAULT 0,
                  admin_response TEXT)''')

    conn.commit()
    conn.close()

init_db()
def retrain_model():
    subprocess.run(["python", "train.py"])

# ==============================
# Create Default Admin
# ==============================
def create_admin():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE username=?", ("admin",))
    if not c.fetchone():
        hashed_password = generate_password_hash("admin123")
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  ("admin", hashed_password))
        conn.commit()

    conn.close()

create_admin()

# ==============================
# Load AI Model
# ==============================
model = tf.keras.models.load_model("model/model.h5")
tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('model/label_encoder.pkl', 'rb'))

with open("intents.json") as file:
    intents = json.load(file)

max_len = 20

# ==============================
# AI Prediction
# ==============================
def predict_class(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_len)
    result = model.predict(padded, verbose=0)[0]

    confidence = float(np.max(result))
    intent = label_encoder.inverse_transform([np.argmax(result)])[0]

    return intent, confidence

# ==============================
# Emotion Detection
# ==============================
def detect_emotion(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.3:
        return "positive"
    elif polarity < -0.3:
        return "negative"
    return "neutral"

# ==============================
# Get Intent Response
# ==============================
def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."
def cluster_unknown_queries():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute("SELECT id, user_message FROM unknown_queries WHERE resolved=0")
    data = c.fetchall()
    conn.close()

    if len(data) < 2:
        return []

    ids = [row[0] for row in data]
    texts = [row[1] for row in data]

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=min(3, len(texts)), random_state=42)
    kmeans.fit(X)

    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        clusters.setdefault(label, []).append((ids[i], texts[i]))

    return clusters

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect('chatbot.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            login_user(User(user[0]))
            return redirect("/admin")

    return render_template("login.html")
@app.route("/retrain", methods=["POST"])
@login_required
def retrain():
    retrain_model()
    return redirect("/admin")
def integrate_new_knowledge():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute("SELECT assigned_tag, user_message, response FROM unknown_queries WHERE resolved=1")
    data = c.fetchall()
    conn.close()

    if not data:
        return

    with open("intents.json", "r") as file:
        intents_data = json.load(file)

    for tag, question, response in data:
        found = False
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                intent["patterns"].append(question)
                intent["responses"].append(response)
                found = True
                break

        if not found:
            intents_data["intents"].append({
                "tag": tag,
                "patterns": [question],
                "responses": [response]
            })

    with open("intents.json", "w") as file:
        json.dump(intents_data, file, indent=4)
@app.route("/retrain", methods=["POST"])
@login_required
def retrain():
    integrate_new_knowledge()
    retrain_model()
    return redirect("/admin")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")

@app.route("/admin")
@login_required
def admin():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM chats")
    total_messages = c.fetchone()[0]

    c.execute("SELECT * FROM chats ORDER BY id DESC LIMIT 10")
    recent_chats = c.fetchall()

    c.execute("SELECT * FROM unknown_queries WHERE resolved=0")
    unresolved = c.fetchall()

    conn.close()
    clusters = cluster_unknown_queries()

    return render_template("admin.html",
                        total_messages=total_messages,
                        recent_chats=recent_chats,
                        unresolved=unresolved,
                        clusters=clusters)


@app.route("/resolve/<int:id>", methods=["POST"])
@login_required
def resolve_query(id):
    new_response = request.form["response"]

    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute("UPDATE unknown_queries SET resolved=1, admin_response=? WHERE id=?",
              (new_response, id))
    conn.commit()
    conn.close()

    return redirect("/admin")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.json["message"]

    # 1️⃣ Check if admin resolved it
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute("SELECT admin_response FROM unknown_queries WHERE user_message=? AND resolved=1", (msg,))
    custom = c.fetchone()
    conn.close()

    if custom:
        return jsonify({"response": custom[0]})

    # 2️⃣ Emotion
    emotion = detect_emotion(msg)

    # 3️⃣ Intent Prediction
    intent, confidence = predict_class(msg)

    # 4️⃣ Low confidence handling
    if confidence < 0.6:
        response = "I'm still learning. An admin will review this question."

        conn = sqlite3.connect('chatbot.db')
        c = conn.cursor()
        c.execute("INSERT INTO unknown_queries (user_message, timestamp) VALUES (?, ?)",
                  (msg, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
    else:
        response = get_response(intent)

    # 5️⃣ Emotion-based tone
    if emotion == "negative":
        response = "I'm sorry you're feeling that way. " + response
    elif emotion == "positive":
        response = "That's great! " + response

    # 6️⃣ Save chat log
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_message, bot_response, emotion, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
              (msg, response, emotion, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return jsonify({"response": response})

# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
