from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import tensorflow as tf
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import traceback

# --- Download NLTK Data ---
nltk.download('stopwords')

# --- Initialize Flask App ---
app = Flask(__name__)

# --- File Paths ---
MODEL_PATH = "model/final_model.keras"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"
ENCODER_TOPIC_PATH = "model/encoder_topic.pkl"
ENCODER_USERNAME_PATH = "model/encoder_username.pkl"

# --- Load Pickle Utility ---
def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

# --- Load Model and Resources ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Keras model not found at: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
vectorizer = load_pickle(VECTORIZER_PATH)
encoder_topic = load_pickle(ENCODER_TOPIC_PATH)
encoder_username = load_pickle(ENCODER_USERNAME_PATH)

# --- Print Number of Classes ---
print(f"✅ Loaded topic encoder with {len(encoder_topic.classes_)} classes.")
print(f"✅ Loaded username encoder with {len(encoder_username.classes_)} classes.")

# --- Preprocessing Utilities ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", '', text)  # URLs
    text = re.sub(r"@\w+", '', text)                   # Mentions
    text = re.sub(r"#\w+", '', text)                   # Hashtags
    text = re.sub(r"[^\w\s\u0900-\u097F]", '', text)   # Special characters
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

def stemming(text):
    tokens = text.split()
    return " ".join([stemmer.stem(word) for word in tokens])

# --- Safe Inverse Transform ---
def safe_inverse_transform(encoder, index):
    if 0 <= index < len(encoder.classes_):
        return encoder.inverse_transform([index])[0]
    return f"Unknown (index {index})"

# --- Prediction Function ---
def predict(text, top_k=5):
    # Preprocess
    cleaned = clean_text(text)
    stemmed = stemming(cleaned)

    # Vectorize
    bow_vector = vectorizer.transform([stemmed]).toarray().astype(np.float32)

    # Predict using Keras model
    predictions = model.predict(bow_vector)
    topic_probs, username_probs = predictions
    topic_probs = topic_probs[0]
    username_probs = username_probs[0]

    # Exact predictions
    topic_index = int(np.argmax(topic_probs))
    username_index = int(np.argmax(username_probs))
    topic_pred = safe_inverse_transform(encoder_topic, topic_index)
    username_pred = safe_inverse_transform(encoder_username, username_index)
    topic_conf = round(float(topic_probs[topic_index]) * 100, 2)
    username_conf = round(float(username_probs[username_index]) * 100, 2)

    # Top-k predictions
    top_topic_indices = np.argsort(topic_probs)[::-1][:top_k]
    top_username_indices = np.argsort(username_probs)[::-1][:top_k]

    top_topics = [
        {
            "topic": safe_inverse_transform(encoder_topic, idx),
            "confidence_percent": round(float(topic_probs[idx]) * 100, 2)
        }
        for idx in top_topic_indices
    ]

    top_usernames = [
        {
            "username": safe_inverse_transform(encoder_username, idx),
            "confidence_percent": round(float(username_probs[idx]) * 100, 2)
        }
        for idx in top_username_indices
    ]

    return {
        "exact_topic": topic_pred,
        "exact_topic_confidence_percent": topic_conf,
        "exact_username": username_pred,
        "exact_username_confidence_percent": username_conf,
        "top_topics": top_topics,
        "top_usernames": top_usernames
    }

# --- API Endpoints ---
@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text or not text.strip():
            return jsonify({"error": "No text provided"}), 400

        result = predict(text)
        return jsonify(result)

    except Exception as e:
        print(traceback.format_exc())  # Optional: log detailed traceback
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "running"})

# --- Run Flask App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
