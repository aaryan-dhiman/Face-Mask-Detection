import os
import sys

# Add root directory to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow Logs and oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import load_data, split_data, GRID_SIZE
from src.model import yolo_loss
from src.deployment_utils import decode_predictions
from tensorflow.keras.models import load_model

def visualize_with_tf():
    """
    Demonstrates using tf.image.draw_bounding_boxes() as requested.
    """
    print("Loading Data...")
    imgs, targets = load_data("data")
    _, _, _, _, X_test, _ = split_data(imgs, targets)
    
    # Take a batch of 4 images
    batch_imgs = X_test[:4]
    
    print("Loading Model...")
    model = load_model("models/mask_detector_final.keras", custom_objects={'yolo_loss': yolo_loss})
    preds = model.predict(batch_imgs, verbose=0)
    
    # Needs format (batch, num_boxes, 4) -> (ymin, xmin, ymax, xmax)
    # tf.image.draw_bounding_boxes expects 0-1 float coords in [y_min, x_min, y_max, x_max]
    
    formatted_boxes_batch = []
    
    for i in range(len(batch_imgs)):
        res = decode_predictions(preds[i], conf_thresh=0.3, iou_thresh=0.4)
        
        img_boxes = []
        if len(res) == 0:
            pass
            
        for r in res:
            box = r['box'] # xmin, ymin, xmax, ymax
            # Convert to ymin, xmin, ymax, xmax
            tf_box = [box[1], box[0], box[3], box[2]]
            img_boxes.append(tf_box)
            
        formatted_boxes_batch.append(img_boxes)
        
    # Pad to same number of boxes for tensor conversion
    max_len = max([len(b) for b in formatted_boxes_batch], default=0)
    if max_len == 0:
        print("No predictions to visualize.")
        return

    tensor_boxes = []
    for b in formatted_boxes_batch:
        padding = max_len - len(b)
        padded = b + [[0.0, 0.0, 0.0, 0.0]] * padding
        tensor_boxes.append(padded)
        
    tensor_boxes = tf.constant(tensor_boxes, dtype=tf.float32) # (Batch, N, 4)
    
    # Images need to be converted to 0-1 for visualization if they are -1..1
    # MobileNet preprocess is -1 to 1.
    imgs_viz = (batch_imgs + 1) / 2.0
    
    # Draw
    output = tf.image.draw_bounding_boxes(imgs_viz, tensor_boxes, colors=np.array([[1.0, 0.0, 0.0]])) # Red boxes
    
    # Save output
    output_np = output.numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axes[i].imshow(output_np[i])
        axes[i].axis('off')
        axes[i].set_title("TF Viz req")
        
    plt.savefig("outputs/tf_draw_boxes_output.png")
    print("Saved outputs/tf_draw_boxes_output.png")

if __name__ == "__main__":
    visualize_with_tf()
