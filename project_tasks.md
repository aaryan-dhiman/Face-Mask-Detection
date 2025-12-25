# Face Mask Detection Project

- [x] **Step 1: Data Loading & Parsing**
    - [x] Parse Pascal VOC XML annotations from `annotations/`
    - [x] Load images from `images/`
    - [x] Visualize sample data (image + box + label)
- [x] **Step 2: Preprocessing**
    - [x] Resize images to uniform size (224x224)
    - [x] Normalize pixel values (0-1 or -1 to 1)
    - [x] Encode labels (with_mask, without_mask, mask_weared_incorrect)
    - [x] Normalize bounding boxes relative to image size
- [x] **Step 3: Dataset Splitting**
    - [x] Split into Train (80%), Validation (10%), Test (10%)
- [x] **Step 4: Model Implementation**
    - [x] Build MobileNetV2 based model
    - [x] Add Classification Head (Softmax)
    - [x] Add Bounding Box Regression Head (Sigmoid/Linear)
    - [x] define Losses (Categorical CrossEntropy + MSE)
- [x] **Step 5: Training**
    - [x] Implement training pipeline (Data Augmentation included)
    - [x] Train model and monitor loss/accuracy
- [x] **Step 6: Evaluation**
    - [x] Calculate Accuracy per class
    - [x] Calculate IoU (Intersection over Union)
    - [x] Visualizing predictions on test set (10 random images)
- [x] **Step 7: Deployment**
    - [x] Create TF Visualization script (`tf_viz.py`) <!-- id: 8 -->
- [x] Finalize End-to-End Pipeline (`run_pipeline.py`) <!-- id: 9 -->

## Phase 2: Model Optimization (Next Session)
- [ ] **Improve Model Accuracy** <!-- id: 10 -->
    - [ ] Increase training epochs (currently low for testing)
    - [ ] Unfreeze MobileNetV2 layers for Fine-Tuning
    - [ ] Experiment with Class Weights or Focal Loss
- [ ] **Data Improvements** <!-- id: 11 -->
    - [ ] Verify dataset balance (Mask vs No Mask)
    - [ ] Add more aggressive data augmentation (Rotation, Zoom)
    - [x] Implement upload/camera interface
    - [x] Display real-time/static inference results
    - [x] Quantize with TFLite
