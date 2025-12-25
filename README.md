# Face Mask Detection System

A complete Deep Learning pipeline to detect face masks using MobileNetV2 + YOLO-style detection.

## Structure
- **`app.py`**: Streamlit Web Application (Image Upload & Detection).
- **`train.py`**: Main training script.
- **`webcam_app.py`**: Real-time Browser-based Webcam Demo.
- **`run_pipeline.py`**: End-to-End Orchestration script (Train -> Eval -> TFLite).
- **`src/`**: Core library code (`dataset`, `model`, `utils`).
- **`scripts/`**: Utility scripts (`evaluate.py`, `tf_viz.py`, `convert_to_tflite.py`).
- **`models/`**: Saved models (.keras, .tflite).
- **`data/`**: Dataset directory.

## Quick Start
1. **Run the Web App**:
   ```bash
   streamlit run app.py
   ```
2. **Run Webcam Demo**:
   ```bash
   streamlit run webcam_app.py
   ```
3. **Retrain Model**:
   ```bash
   python train.py
   ```
