import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

    # Function to send email
def send_email(to_email, subject, body):
    # Update with your email credentials
    sender_email = 'jdjoshkd@gmail.com'
    sender_password ='mygg gwhe teeu pock'

    # Set up the MIME
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = to_email
    message['Subject'] = subject

    # Attach the body of the email
    message.attach(MIMEText(body, 'plain'))

    # Connect to the SMTP server (Gmail)
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, message.as_string())


# Load the saved CNN model
cnn_model = load_model("https://github.com/basilll007/sample_app/blob/main/cnn_model.h5")

# Define the classes
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Streamlit app
st.title("CNN Image Classifier")

# Get user's email input
#user_email = st.text_input("Enter your email:")

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
    # Send email with prediction results
    email_to_send = "arularulak96@gmail.com"  # Update with the recipient's email
    email_subject = "Image Classification Prediction"
    email_body = f"Predicted Class: {predicted_class}\nAccuracy: {accuracy:.2f}%"
    
    try:
        send_email(email_to_send, email_subject, email_body)
        st.success(f"Email sent successfully to {email_to_send}")
    except Exception as e:
        st.error(f"Error sending email: {e}")



