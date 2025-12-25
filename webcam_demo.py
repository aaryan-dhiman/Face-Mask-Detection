import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from src.model import yolo_loss
from src.deployment_utils import decode_predictions, draw_boxes
import subprocess
import sys

def main():
    print("Loading model...")
    try:
        model = load_model("models/mask_detector_final.keras", custom_objects={'yolo_loss': yolo_loss})
    except:
        print("Model not found. Run train.py first.")
        return

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Webcam Inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess for Model
        # Resize to 224x224
        input_frame = cv2.resize(frame, (224, 224))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        input_frame = img_to_array(input_frame)
        input_frame = preprocess_input(input_frame)
        input_frame = np.expand_dims(input_frame, axis=0) # (1, 224, 224, 3)

        # Predict
        preds = model.predict(input_frame, verbose=0)[0] # (7, 7, 8)
        
        # Decode
        # Lower threshold for realtime demo flow
        results = decode_predictions(preds, conf_thresh=0.5, iou_thresh=0.4)
        
        # Draw on Original Frame
        # Boxes are normalized 0-1, so they scale to any image size
        frame = draw_boxes(frame, results)
       # Show Result
        try:
            cv2.imshow("Face Mask Detection", frame)
        except cv2.error as e:
            print("\n[INFO] Local window not supported. Automatically launching Browser Interface...")
            # Close camera before switching
            cap.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass
                
            # Launch Streamlit
            subprocess.run(["streamlit", "run", "webcam_app.py"])
            return

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

if __name__ == "__main__":
    main()
