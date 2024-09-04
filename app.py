import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    return model

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img) / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the digit
def predict_digit(model, img):
    predictions = model.predict(img)
    digit = np.argmax(predictions, axis=1)[0]
    return digit

# Streamlit App
st.title("Digit Classification App")

model = load_trained_model()

uploaded_file = st.file_uploader("Upload an image of a digit...", type=["jpg", "png"])


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify Image'):
        processed_image = preprocess_image(img)
        digit = predict_digit(model, processed_image)
        st.write(f"The digit in the image is: {digit}")
