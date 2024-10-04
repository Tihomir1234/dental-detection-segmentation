import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Cache the model loading process
@st.cache_resource
def load_model():
    print(Path.cwd())
    model_path = Path("/mount/src/dental-detection-segmentation/best.pt")
    model = YOLO(model_path)
    return model

# Function to run prediction on an image
def predict(image, model):
    img_array = np.array(image)  # Convert Pillow image to NumPy array
    img_rgb = img_array  # Image is already in RGB mode when opened with Pillow
    results = model([img_rgb])  # Run the YOLO model on the image
    return results

# Display the prediction results using Matplotlib
def display_results(results):
    fig, ax = plt.subplots()
    plt.imshow(results[0].plot()[..., ::-1])  # Plot the results and reverse channels for correct display
    plt.axis('off')
    st.pyplot(fig)

# Load the model
model = load_model()

# Streamlit app title
st.title("Dental Disease Detection")

# Upload an image
uploaded_file = st.file_uploader("Upload a dental image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)  # Load image using PIL
    results = predict(image, model)  # Predict with YOLO model
    display_results(results)  # Show the results
