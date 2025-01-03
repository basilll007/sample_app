import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import streamlit as st
from PIL import Image

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Streamlit app
def main():
    st.title("Image Classification using MobileNetV2")
    st.write("Upload an image to classify it.")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open the uploaded image using PIL (Python Imaging Library)
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        
        # Preprocess the image for the model
        img = img.resize((224, 224))  # Resize image to (224, 224) for MobileNetV2
        img_array = np.array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess image (this is specific to MobileNetV2)
        
        # Predict the class
        prediction = model.predict(img_array)
        decoded_predictions = decode_predictions(prediction, top=3)[0]  # Get top 3 predictions
        
        st.write("Top Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i+1}. {label}: {score*100:.2f}% confidence")
        
if __name__ == "__main__":
    main()
