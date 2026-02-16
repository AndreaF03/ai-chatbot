from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
import random

lemmatizer = WordNetLemmatizer()
model = tf.keras.models.load_model('model.h5')

with open('intents.json') as file:
    intents = json.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.json["message"]
    return jsonify({"response": get_response(msg)})

def get_response(message):
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in message.lower():
                return random.choice(intent["responses"])
    return "Sorry, I don't understand."

if __name__ == "__main__":
    app.run(debug=True)
