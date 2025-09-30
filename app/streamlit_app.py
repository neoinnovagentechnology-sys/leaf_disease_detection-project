import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🌿 Leaf Disease Detection")
st.write("Upload a leaf image to classify its disease.")

# Load Model
model = tf.keras.models.load_model(r"C:\Users\naren\Downloads\leaf_disease_detection\models\plant_model.h5")
class_names = ["Class1", "Class2", "Class3"]  # Replace with dataset classes

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    img = image.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {pred_class} ({confidence:.2f}% confidence)")
