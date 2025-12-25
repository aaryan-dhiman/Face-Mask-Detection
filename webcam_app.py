import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from src.model import yolo_loss
from src.deployment_utils import decode_predictions, draw_boxes

# Config
MODEL_PATH = "models/mask_detector_final.keras"

@st.cache_resource
def load_detection_model():
    return load_model(MODEL_PATH, custom_objects={'yolo_loss': yolo_loss})

def main():
    st.title("Live Face Mask Detection")
    st.write("Use your webcam to detect masks in real-time.")

    model = load_detection_model()

    # Streamlit Camera Input
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Preprocess
        original_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(original_img, (224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        
        # Decode
        results = decode_predictions(preds[0], conf_thresh=0.5, iou_thresh=0.4)
        
        # Draw
        result_img = draw_boxes(original_img, results)
        
        st.image(result_img, caption="Detection Result", use_container_width=True)

if __name__ == "__main__":
    main()
