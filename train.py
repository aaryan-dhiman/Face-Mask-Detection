import os
import tensorflow as tf
import numpy as np
from src.dataset import load_data, split_data, IMAGE_SIZE, GRID_SIZE
from src.model import create_model, yolo_loss
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Parameters
BATCH_SIZE = 32
EPOCHS = 15 # Updated to 15 as requested
LR = 1e-4

def augment(image, target):
    # image: (224, 224, 3)
    # target: (7, 7, 8)
    
    # Random brightness/contrast/saturation
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    
    # Random Flip Left Right
    # Need to flip target too
    
    # 50% chance
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        
        # Flip Target
        # Target shape: (7, 7, 8)
        # Flip width (axis 1)
        target = tf.reverse(target, axis=[1])
        
        # Adjust x-coordinate (index 1)
        # x_new = 1.0 - x_old
        # Only where object exists (index 0 == 1)
        
        # Create a mask for objectness
        obj_mask = target[..., 0:1] # (7, 7, 1)
        
        # Extract x
        x = target[..., 1:2]
        
        # Flip x
        x_flipped = 1.0 - x
        
        # Update target
        # We need to construct the new target tensor
        # target indices: 0:obj, 1:x, 2:y, 3:w, 4:h, 5..:cls
        
        # We can just update the slices using concat
        # Note: tf.concat needs matching dims
        
        t0 = target[..., 0:1] # obj
        t1 = x_flipped        # x (flipped)
        t2 = target[..., 2:8] # y, w, h, cls (unchanged)
        
        # However, we only apply 1-x where obj exists?
        # Actually 1-0 = 1, so empty cells will have x=1. 
        # But obj is 0, so loss ignores it. This is safe!
        
        target = tf.concat([t0, t1, t2], axis=-1)
        
    return image, target

def unaugment(image, target):
    return image, target

def create_dataset(X, y, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if is_train:
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(unaugment, num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    print("Loading data...")
    dataset_dir = "data"
    images, targets = load_data(dataset_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(images, targets)
    
    print(f"Train samples: {len(X_train)}")
    
    train_ds = create_dataset(X_train, y_train, is_train=True)
    val_ds = create_dataset(X_val, y_val, is_train=False)

    # Create Model
    model = create_model(input_shape=IMAGE_SIZE + (3,), num_classes=3)
    
    # --- STAGE 1: Train Head ---
    # --- STAGE 1: Train Head ---
    if os.path.exists("models/mask_detector_stage1.keras"):
        print("\n[Stage 1] Checkpoint found! Loading weights and skipping Stage 1...")
        model.load_weights("models/mask_detector_stage1.keras")
    else:
        print("\n[Stage 1] Training Head (Frozen Backbone)...")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss=yolo_loss)
        
        checkpoint = ModelCheckpoint("models/mask_detector_stage1.keras", monitor='val_loss', save_best_only=True, verbose=1)
        
        history_1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=5, # Stage 1: 5 epochs
            callbacks=[checkpoint]
        )
    
    # --- STAGE 2: Fine-Tuning ---
    print("\n[Stage 2] Fine-Tuning (Unfreezing top layers)...")
    
    # Unfreeze the base model
    base_model = model.layers[1] # MobileNetV2 is usually the 2nd layer (index 1) after Input, or index 0 if Functional? 
    # Let's check: Model(inputs=base_model.input...) -> Input is layer 0, or base_model is layer 1. 
    # Debug: Print layers
    # Actually, in Functional API, layers are graph nodes. 
    # To unfreeze, we access the MobileNetV2 layer object directly if we can, or just iterate.
    # We wrapped it: base_model = MobileNetV2(...)
    # But 'model' is a new Model. The 'base_model' variable is lost. 
    # We need to find the MobileNetV2 layer within 'model'.
    
    # However, 'base_model' was used to construct the graph. It might not be a single "Layer" in 'model.layers' if we used its output directly.
    # Ah, MobileNetV2(...) returns a Model object. When we use it like `x = base_model.output`, the layers of base_model become part of the graph?
    # No, typically Keras flattens it effectively or treats it as shared layers.
    # To be safe in Functional API without a nested 'layer', we iterate and unfreeze.
    
    # Let's rely on model.layers. MobileNetV2 layers are many.
    # We want to unfreeze the last N layers.
    
    model.trainable = True # Unfreeze everything
    
    # Freeze all except last 30 layers
    # Total layers ~ 155 for MobileNetV2
    FINE_TUNE_AT = 120
    
    for layer in model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
        
    # Recompile with Low LR
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=yolo_loss)
    model.summary()
    
    checkpoint_final = ModelCheckpoint("models/mask_detector_final.keras", monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10, # Stage 2: 10 epochs
        callbacks=[checkpoint_final, early_stop]
    )
    
    model.save("models/mask_detector_final.keras")
    print("Training Complete.")

if __name__ == "__main__":
    main()
