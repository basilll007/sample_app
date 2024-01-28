import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the saved CNN model
cnn_model = load_model("cnn_model.h5")

# Define the classes
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Streamlit app
st.title("Image Classifier")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for the model
    img_array = np.array(image.resize((32, 32)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction using the loaded CNN model
    prediction = cnn_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    accuracy = prediction[0][predicted_class_index] * 100

    # Display the predicted class and accuracy
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Accuracy: {accuracy:.2f}%")
