import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import requests

# ==============================
# Load trained model
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/plant_model.h5")

model = load_model()

# ==============================
# Load class labels
# ==============================
with open("models/class_labels.json", "r") as f:
    class_names = json.load(f)

# ==============================
# AI Remedy Suggestion (Groq API)
# ==============================
def get_ai_remedy(disease_name: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    # Use Streamlit secrets for API key
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert plant pathologist. Provide short, actionable remedies in simple terms."},
            {"role": "user", "content": f"Suggest remedies for {disease_name} disease in plants."}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ API Error: {str(e)}"

# ==============================
# Streamlit UI
# ==============================
st.title("🌱 Plant Disease Detection + AI Remedies")

uploaded_file = st.file_uploader("📷 Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    disease_name = class_names.get(str(class_idx), "Unknown Disease")

    # Show results
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)
    st.success(f"✅ Prediction: **{disease_name}** ({confidence:.2f}% confidence)")

    # AI-based Remedy
    with st.spinner("💡 Fetching AI Remedy Suggestion..."):
        remedy = get_ai_remedy(disease_name)
    st.subheader("🌿 Suggested Remedies")
    st.write(remedy)
