import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('best_model.h5')

# Function to predict glasses or no glasses
def predict_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "With Glasses" if prediction[0][0] > 0.5 else "Without Glasses"

# Streamlit UI
st.title("Glasses Image Classification")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = predict_image(image)
    st.write(f"Prediction: **{prediction}**")
