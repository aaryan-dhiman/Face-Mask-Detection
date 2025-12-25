import subprocess
import sys
import os

def run_step(script_name, description):
    print(f"\n{'='*50}")
    print(f"PIPELINE STEP: {description}")
    print(f"{'='*50}\n")
    
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline failed at step: {script_name}")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] Completed: {script_name}")

def main():
    print("Starting End-to-End Face Mask Detection Pipeline...")
    
    # Step 1: Data Check (Implicit in training, but let's check dataset.py runs)
    # We won't run it fully as it just loads data, but we can assume if train works, data works.
    
    # Step 2: Training
    if not os.path.exists("models/mask_detector_final.keras"):
        print("Model not found. Starting Training...")
        run_step("train.py", "Model Training (with Augmentation)")
    else:
        print("Model found. Skipping Training (Remove file to Retrain).")
        # Optional: Force retrain? User might want quick check.
        # Let's prompt or just proceed. For a pipeline, usually reliable reproducibility is key.
        # We will assume if user runs pipeline, they might want to ensure everything is fresh? 
        # For now, let's just run train.py which will overwrite.
        # actually, training takes time. Let's ask or just run. 
        # Given user's CPU constraints, maybe skip? 
        # No, a pipeline *runs* the process. I'll print a message.
        pass

    # Step 3: Evaluation
    run_step("scripts/evaluate.py", "Model Evaluation & Visualization")
    
    # Step 4: TFLite Conversion (Deployment Prep)
    run_step("scripts/convert_to_tflite.py", "Quantization & TFLite Conversion")
    
    # Step 5: Verification (TF Viz)
    run_step("scripts/tf_viz.py", "TF.Image Visualization Check")
    
    print(f"\n{'='*50}")
    print("PIPELINE COMPLETE.")
    print("1. Web App ready: Run 'streamlit run app.py'")
    print("2. Webcam Demo ready: Run 'python webcam_demo.py'")
    print("3. TFLite Model ready: models/mask_detector.tflite")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
