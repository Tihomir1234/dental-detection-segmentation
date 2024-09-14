import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    print(Path.cwd())
    model_path = Path("/mount/src/dental-detection-segmentation/best.pt")

    model = YOLO(model_path)
    return model

def predict(image, model):
    img_array = np.array(image)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    results = model([img_rgb])
    return results

def display_results(results):
    fig, ax = plt.subplots()
    plt.imshow(results[0].plot()[..., ::-1])
    plt.axis('off')
    st.pyplot(fig)

model = load_model()

st.title("Dental Disease Detection")

uploaded_file = st.file_uploader("Upload a dental image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    results = predict(image, model)
    display_results(results)
