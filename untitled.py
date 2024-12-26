import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2

# Define constants
MODEL_PATH = r'C:\Users\hengb\Downloads\FACIAL EXPRESSIONS\facial_expression_model.h5'  # Update to your model's path
IMAGE_SIZE = (48, 48)  # Input size expected by the model
EXPRESSIONS = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']  # Define labels

# Load the model
@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)

model = load_emotion_model()

# Preprocess the uploaded image
def preprocess_image(image):
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    # Resize to model's input size
    image = image.resize(IMAGE_SIZE)
    # Normalize pixel values
    image_array = np.array(image) / 255.0
    # Add batch and channel dimensions
    return np.expand_dims(np.expand_dims(image_array, axis=-1), axis=0)

# Predict emotion
def predict_emotion(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    confidence = np.max(predictions) * 100
    predicted_label = EXPRESSIONS[np.argmax(predictions)]
    return predicted_label, confidence

# Streamlit app
st.title("Facial Emotion Detection")
st.write("Upload an image, and the model will predict the emotion along with the confidence score.")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict and display results
    with st.spinner("Analyzing the image..."):
        predicted_label, confidence = predict_emotion(image)

    st.success(f"Prediction: {predicted_label} ({confidence:.2f}% confidence)")

    # Overlay prediction on the image
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay_text = f"{predicted_label} ({confidence:.2f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(overlay_text, font, font_scale, thickness)[0]
    text_x = (image_cv.shape[1] - text_size[0]) // 2
    text_y = 50
    cv2.putText(image_cv, overlay_text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness)

    # Show the image with overlay
    st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption="Predicted Image", use_column_width=True)
