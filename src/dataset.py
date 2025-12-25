import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Config
IMAGE_SIZE = (224, 224)
GRID_SIZE = 7
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]
CLASS_MAP = {k: v for v, k in enumerate(CLASSES)}

def load_data(dataset_dir):
    images = []
    targets = []
    
    annotations_dir = os.path.join(dataset_dir, "annotations")
    images_dir = os.path.join(dataset_dir, "images")

    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]
    print(f"Found {len(xml_files)} annotation files.")

    for xml_file in xml_files:
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        
        filename = root.find("filename").text
        img_path = os.path.join(images_dir, filename)
        
        # Handle file extension issues
        if not os.path.exists(img_path):
            base = os.path.splitext(filename)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                if os.path.exists(os.path.join(images_dir, base + ext)):
                    img_path = os.path.join(images_dir, base + ext)
                    break
        
        if not os.path.exists(img_path):
            continue

        # Load sizes
        size = root.find("size")
        w_orig = int(size.find("width").text)
        h_orig = int(size.find("height").text)

        # Load Image
        image = cv2.imread(img_path)
        if image is None: 
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        image = img_to_array(image)
        image = preprocess_input(image)

        # Build Target Grid (7, 7, 8)
        # 0: Objectness (1 if object), 1-4: x,y,w,h (norm), 5-7: Classes
        target_grid = np.zeros((GRID_SIZE, GRID_SIZE, 8))

        boxes_found = False
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASS_MAP:
                continue
            
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Normalize 0-1
            cx = (xmin + xmax) / 2.0 / w_orig
            cy = (ymin + ymax) / 2.0 / h_orig
            w_norm = (xmax - xmin) / w_orig
            h_norm = (ymax - ymin) / h_orig

            # Grid Cell
            col = int(cx * GRID_SIZE)
            row = int(cy * GRID_SIZE)
            
            if col >= GRID_SIZE: col = GRID_SIZE - 1
            if row >= GRID_SIZE: row = GRID_SIZE - 1

            # Assign to grid (last object overwrites)
            target_grid[row, col, 0] = 1.0
            
            # GRID RELATIVE COORDINATES
            # cx, cy are normalized 0-1
            # We want offset within the cell 0-1
            # x_offset = cx * GRID_SIZE - col
            # y_offset = cy * GRID_SIZE - row
            
            x_offset = cx * GRID_SIZE - col
            y_offset = cy * GRID_SIZE - row
            
            target_grid[row, col, 1:5] = [x_offset, y_offset, w_norm, h_norm]
            target_grid[row, col, 5 + CLASS_MAP[name]] = 1.0
            boxes_found = True

        if boxes_found:
            images.append(image)
            targets.append(target_grid)

    return np.array(images, dtype="float32"), np.array(targets, dtype="float32")

def split_data(images, targets):
    # Random split (Stratified is hard with grids)
    X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    imgs, tgs = load_data("d:/Aaryan_DL_Lab_Assessment_Exam")
    print(f"Images: {imgs.shape}")
    print(f"Targets: {tgs.shape}")
