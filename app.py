import streamlit as st
import sys
import os

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from src.deployment_utils import decode_predictions, draw_boxes
from src.model import yolo_loss

# Config
MODEL_PATH = "models/mask_detector_final.keras"

@st.cache_resource
def load_detection_model():
    # Helper to load model with custom object
    try:
        if tf.io.gfile.exists(MODEL_PATH):
            model = load_model(MODEL_PATH, custom_objects={'yolo_loss': yolo_loss})
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

st.title("Face Mask Detection System")
st.write("Upload an image to detect if people are wearing masks.")

model = load_detection_model()

if model is None:
    st.warning("Model not found yet. Please wait for training to complete.")
else:
    st.success("Model loaded successfully!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("Detecting...")
    
    # Preprocess
    img_array = np.array(image)
    orig_h, orig_w = img_array.shape[:2]
    
    resized = cv2.resize(img_array, (224, 224))
    x = img_to_array(resized)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    
    # Predict
    if model:
        pred_grid = model.predict(x)[0] # (7, 7, 8)
        
        # Decode
        results = decode_predictions(pred_grid, conf_thresh=0.5, iou_thresh=0.4)
        
        # Draw
        # We need to draw on the Original size image.
        out_img = np.array(image)
        out_img = draw_boxes(out_img, results)
        
        st.image(out_img, caption='Processed Image', use_container_width=True)
        
        # Metrics
        st.write(f"Found {len(results)} faces.")
        for res in results:
            st.write(f"- {res['label']} ({res['conf']:.2f})")
