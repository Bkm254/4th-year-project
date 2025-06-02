import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the trained model
model = load_model('parasite_classifier_model.h5')

# Class labels (replace with your actual class names)
classes = ['Fasciola', 'haemonchus', 'strongyloides']

st.title("Parasite Image Classifier")
st.write("Upload an image of a parasite egg to classify it.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image with target size matching training
    img = load_img(uploaded_file, target_size=(150, 150))
    # Convert to array and normalize
    img_array = img_to_array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict class probabilities
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
