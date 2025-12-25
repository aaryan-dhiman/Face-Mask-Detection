import numpy as np
import tensorflow as tf
import os
import cv2
import sys

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_data, split_data, GRID_SIZE
from src.model import yolo_loss
from src.deployment_utils import decode_predictions, draw_boxes, CLASSES
from tensorflow.keras.models import load_model

def calculate_iou(box1, box2):
    # box: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def evaluate():
    print("Loading Data for Evaluation...")
    imgs, targets = load_data("d:/Aaryan_DL_Lab_Assessment_Exam/data")
    _, _, _, _, X_test, y_test = split_data(imgs, targets)
    
    print(f"Test Set: {len(X_test)} samples")
    
    print("Loading Model...")
    try:
        model = load_model("models/mask_detector_final.keras", custom_objects={'yolo_loss': yolo_loss})
    except:
        print("Model not found.")
        return
    
    metrics = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in CLASSES}
    total_iou = 0
    total_matches = 0
    
    batch_size = 32
    preds = model.predict(X_test, batch_size=batch_size)
    
    # Visualization: 10 random images
    indices = np.random.choice(len(X_test), 10, replace=False)
    save_dir = "outputs/test_results"
    os.makedirs(save_dir, exist_ok=True)
    
    # Loop for Metrics
    for i in range(len(X_test)):
        # Decode GT
        gt_grid = y_test[i]
        gt_boxes = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if gt_grid[r, c, 0] == 1:
                    # GT is now offset, need to decode to global for IoU
                    x_off, y_off, w, h = gt_grid[r, c, 1:5]
                    cx = (c + x_off) / GRID_SIZE
                    cy = (r + y_off) / GRID_SIZE
                    # cx, cy, w, h = gt_grid[r, c, 1:5] # OLD
                    cls_idx = np.argmax(gt_grid[r, c, 5:])
                    gt_boxes.append({
                        'box': [cx-w/2, cy-h/2, cx+w/2, cy+h/2],
                        'class': CLASSES[cls_idx]
                    })

        # Decode Pred
        # DEBUG: Print max conf
        max_conf = np.max(preds[i][..., 0])
        if i == 0:
            print(f"Sample 0 Max Conf: {max_conf}")
            
        pred_res = decode_predictions(preds[i], conf_thresh=0.25, iou_thresh=0.4)
        
        # Save Visualization for selected indices
        if i in indices:
            # Denormalize image for saving
            # Image was Preprocessed (MobileNet -1..1 or 0..1?) 
            # MobileNetV2 preprocess_input maps to -1..1
            img_vis = ((X_test[i] + 1) * 127.5).astype(np.uint8)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            
            # Draw Preds
            img_vis = draw_boxes(img_vis, pred_res)
            
            # Draw GT (Blue) - Optional, just showing Preds as per prompt
            cv2.imwrite(os.path.join(save_dir, f"res_{i}.jpg"), img_vis)

        # Match for Metrics
        matched_gt = set()
        for p in pred_res:
            best_iou = 0
            best_gt_idx = -1
            p_box = p['box']
            p_cls = p['label']
            
            for j, g in enumerate(gt_boxes):
                if j in matched_gt: continue
                iou = calculate_iou(p_box, g['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= 0.5 and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                total_iou += best_iou
                total_matches += 1
                gt_cls = gt_boxes[best_gt_idx]['class']
                if p_cls == gt_cls:
                    metrics[p_cls]['tp'] += 1
                else:
                    metrics[p_cls]['fp'] += 1
            else:
                metrics[p_cls]['fp'] += 1
        
        for j, g in enumerate(gt_boxes):
            if j not in matched_gt:
                metrics[g['class']]['fn'] += 1
                
    print("\nEvaluation Results (IoU=0.5):")
    print("-" * 30)
    for cls in CLASSES:
        m = metrics[cls]
        tp, fp, fn = m['tp'], m['fp'], m['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Class: {cls}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
    
    avg_iou = total_iou / total_matches if total_matches > 0 else 0
    print("-" * 30)
    print(f"Average IoU (on matched boxes): {avg_iou:.4f}")
    print(f"Visualizations saved to {save_dir}/")

if __name__ == "__main__":
    evaluate()
