import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load class labels
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# Load trained model
model = tf.keras.models.load_model("models/plant_model.h5")

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to identify the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    class_idx = str(np.argmax(preds))
    disease_name = class_labels[class_idx]

    st.success(f"Prediction: **{disease_name}**")
