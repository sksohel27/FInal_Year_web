import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Download NLTK Data ---
nltk.download('stopwords')

# --- File Paths ---
MODEL_PATH = "model/final_model.keras"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"
ENCODER_TOPIC_PATH = "model/encoder_topic.pkl"
ENCODER_USERNAME_PATH = "model/encoder_username.pkl"

# --- Load Pickle Utility ---
@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# --- Load Resources ---
@st.cache_resource
def load_model_and_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    vectorizer = load_pickle(VECTORIZER_PATH)
    encoder_topic = load_pickle(ENCODER_TOPIC_PATH)
    encoder_username = load_pickle(ENCODER_USERNAME_PATH)
    return model, vectorizer, encoder_topic, encoder_username

model, vectorizer, encoder_topic, encoder_username = load_model_and_resources()

# --- Preprocessing ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"#\w+", '', text)
    text = re.sub(r"[^\w\s\u0900-\u097F]", '', text)
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

def stemming(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

def safe_inverse_transform(encoder, index):
    if 0 <= index < len(encoder.classes_):
        return encoder.inverse_transform([index])[0]
    return f"Unknown (index {index})"

def predict(text, top_k=5):
    cleaned = clean_text(text)
    stemmed = stemming(cleaned)
    bow_vector = vectorizer.transform([stemmed]).toarray().astype(np.float32)

    topic_probs, username_probs = model.predict(bow_vector)
    topic_probs = topic_probs[0]
    username_probs = username_probs[0]

    topic_index = int(np.argmax(topic_probs))
    username_index = int(np.argmax(username_probs))

    topic_pred = safe_inverse_transform(encoder_topic, topic_index)
    username_pred = safe_inverse_transform(encoder_username, username_index)

    top_topic_indices = np.argsort(topic_probs)[::-1][:top_k]
    top_username_indices = np.argsort(username_probs)[::-1][:top_k]

    top_topics = [
        (safe_inverse_transform(encoder_topic, idx), round(float(topic_probs[idx]) * 100, 2))
        for idx in top_topic_indices
    ]

    top_usernames = [
        (safe_inverse_transform(encoder_username, idx), round(float(username_probs[idx]) * 100, 2))
        for idx in top_username_indices
    ]

    return {
        "topic_pred": topic_pred,
        "username_pred": username_pred,
        "topic_conf": round(float(topic_probs[topic_index]) * 100, 2),
        "username_conf": round(float(username_probs[username_index]) * 100, 2),
        "top_topics": top_topics,
        "top_usernames": top_usernames
    }

# --- Streamlit UI ---
st.title("🔍 Text Prediction App")
st.write("Enter a text snippet below to predict the **topic** and **username**.")

user_input = st.text_area("Input Text", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Predicting..."):
            result = predict(user_input)

        st.subheader("🎯 Exact Predictions")
        st.write(f"**Topic:** {result['topic_pred']} ({result['topic_conf']}%)")
        st.write(f"**Username:** {result['username_pred']} ({result['username_conf']}%)")

        st.subheader("📈 Top Topics")
        for topic, conf in result["top_topics"]:
            st.write(f"- {topic} ({conf}%)")

        st.subheader("📈 Top Usernames")
        for username, conf in result["top_usernames"]:
            st.write(f"- {username} ({conf}%)")
