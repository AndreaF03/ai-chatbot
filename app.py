from flask import Flask, render_template, request, jsonify, redirect
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


# ==============================
# Initialize Flask App FIRST
# ==============================
app = Flask(__name__)
app.secret_key = "supersecretkey"   # Required for login sessions


# ==============================
# Initialize Login Manager
# ==============================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ==============================
# User Class
# ==============================
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
                  timestamp TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT)''')

    conn.commit()
    conn.close()


init_db()


# ==============================
# Create Default Admin
# ==============================
def create_admin():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()

    hashed_password = generate_password_hash("admin123")

    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  ("admin", hashed_password))
        conn.commit()
    except:
        pass

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
# Predict Intent
# ==============================
def predict_class(sentence):
    sequence = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(sequence, maxlen=max_len)
    result = model.predict(padded)[0]

    confidence = np.max(result)
    if confidence < 0.6:
        return None

    return label_encoder.inverse_transform([np.argmax(result)])[0]


# ==============================
# Get Bot Response
# ==============================
def get_response(tag):
    if tag:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."


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


@app.route("/admin")
@login_required
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


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.json["message"]

    tag = predict_class(msg)
    response = get_response(tag)

    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_message, bot_response, timestamp) VALUES (?, ?, ?)",
              (msg, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return jsonify({"response": response})


# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
