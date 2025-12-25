import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from src.model import yolo_loss

def convert():
    print("Loading model...")
    try:
        model = tf.keras.models.load_model("models/mask_detector_final.keras", custom_objects={'yolo_loss': yolo_loss})
    except:
        print("Model not found. Please wait for training.")
        return

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Quantization (Dynamic Range)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    with open("models/mask_detector.tflite", "wb") as f:
        f.write(tflite_model)
        
    print("Saved models/mask_detector.tflite")

if __name__ == "__main__":
    convert()
