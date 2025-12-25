import numpy as np
import cv2
import tensorflow as tf

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
COLORS = [(0, 255, 0), (255, 0, 0), (255, 255, 0)] # Green, Red, Cyan

def decode_predictions(grid, conf_thresh=0.5, iou_thresh=0.5):
    """
    Decodes the (7,7,8) grid output into a list of boxes.
    Apply NMS.
    """
    # grid: (7, 7, 8)
    boxes = []
    confidences = []
    class_ids = []

    rows, cols, _ = grid.shape
    
    for r in range(rows):
        for c in range(cols):
            cell = grid[r, c]
            conf = cell[0]
            
            if conf > conf_thresh:
                # cell[1:5] are [x_offset, y_offset, w, h]
                x_off, y_off, w, h = cell[1:5]
                
                # Convert back to Global Normalized cx, cy
                cx = (c + x_off) / 13.0 # GRID_SIZE is 13
                cy = (r + y_off) / 13.0
                
                # Convert to xmin, ymin, xmax, ymax
                xmin = cx - w / 2
                ymin = cy - h / 2
                xmax = cx + w / 2
                ymax = cy + h / 2
                
                # Get class
                # cell[5:] are class probs
                cls_scores = cell[5:]
                cls_id = np.argmax(cls_scores)
                
                boxes.append([xmin, ymin, xmax, ymax])
                confidences.append(float(conf))
                class_ids.append(cls_id)

    indices = cv2.dnn.NMSBoxes(
        bboxes=[[int(b[0]*416), int(b[1]*416), int((b[2]-b[0])*416), int((b[3]-b[1])*416)] for b in boxes], # NMS needs int xywh usually? No, check opencv
        # OpenCV NMSBoxes expects [x, y, w, h] in pixel coords usually.
        # Let's adjust inputs to NMSBoxes.
        scores=confidences,
        score_threshold=conf_thresh,
        nms_threshold=iou_thresh
    )
    
    final_results = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_results.append({
                "box": boxes[i],
                "conf": confidences[i],
                "class_id": class_ids[i],
                "label": CLASSES[class_ids[i]]
            })
            
    return final_results

def draw_boxes(image, results):
    """
    Draw boxes on image.
    image: (H, W, 3) BGR or RGB (uint8)
    results: list of dicts from decode_predictions
    """
    img = image.copy()
    h_img, w_img = img.shape[:2]
    
    for res in results:
        box = res['box']
        label = res['label']
        conf = res['conf']
        cls_id = res['class_id']
        
        xmin = int(box[0] * w_img)
        ymin = int(box[1] * h_img)
        xmax = int(box[2] * w_img)
        ymax = int(box[3] * h_img)
        
        color = COLORS[cls_id]
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return img
