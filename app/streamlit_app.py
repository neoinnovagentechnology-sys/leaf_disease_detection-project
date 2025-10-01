import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import requests

# Load trained model
model = tf.keras.models.load_model("models/plant_model.h5")

# Load class labels
with open("models/class_labels.json") as f:
    class_names = json.load(f)

# Function: AI Remedy Suggestion using Groq (example)
def get_ai_remedy(disease_name: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer YOUR_GROQ_API_KEY"}
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an expert plant pathologist. Suggest remedies for plant diseases in simple terms."},
            {"role": "user", "content": f"Suggest remedies for {disease_name} disease in plants."}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# Streamlit UI
st.title("🌱 Plant Disease Detection + Remedies")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])
if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    disease_name = class_names[class_idx]

    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)
    st.success(f"Prediction: {disease_name} ({confidence:.2f}% confidence)")

    # AI-based Remedy
    st.info("💡 Fetching AI Remedy Suggestion...")
    remedy = get_ai_remedy(disease_name)
    st.write(remedy)
