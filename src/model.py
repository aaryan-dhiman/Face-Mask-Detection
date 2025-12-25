import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Reshape, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def create_model(input_shape=(416, 416, 3), num_classes=3):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False # Start frozen

    x = base_model.output # 7x7x1280
    
    # Improved Detection Head (The "Neck")
    x = Conv2D(256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Output: 5 (Conf, x, y, w, h) + num_classes
    filters = 5 + num_classes
    x = Conv2D(filters, kernel_size=1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

def iou_loss(true_box, pred_box):
    # true_box/pred_box: [x_off, y_off, w, h] (relative to grid cell usually, but for IoU we need comparable units)
    # The x_off, y_off are 0-1 within cell. w, h are normalized 0-1 to image.
    # To compute IoU accurately, we can treat them as just numbers. 
    # However, IoU needs global overlap.
    # But since we are comparing within the same cell hypothesis, we can approximate or decode.
    # Better: Transform to local cell coordinates (x,y w,h all relative to cell or image?)
    # Easier approximation for optimization:
    # Use MSE for x,y (offsets) and MSE for w,h? No, user wants IoU loss.
    # Let's decode to "local cell units".
    # x_center = true_box[..., 0]
    # y_center = true_box[..., 1]
    # w = true_box[..., 2] * 10.0 (GRID_SIZE) -> to make it comparable to x_center domain 0-1
    # Actually, simplest GIoU/IoU implementation works on min/max.
    
    # Input is [x_off, y_off, w_norm, h_norm]
    # x_off, y_off \in [0, 1]
    # w_norm, h_norm \in [0, 1] relative to image
    
    # Let's convert validly:
    # Scale w, h by GRID_SIZE to match x_off, y_off scale?
    from src.dataset import GRID_SIZE
    GRID_SIZE = float(GRID_SIZE)
    
    xt = true_box[..., 0]
    yt = true_box[..., 1]
    wt = true_box[..., 2] * GRID_SIZE
    ht = true_box[..., 3] * GRID_SIZE
    
    xp = pred_box[..., 0]
    yp = pred_box[..., 1]
    wp = pred_box[..., 2] * GRID_SIZE
    hp = pred_box[..., 3] * GRID_SIZE
    
    # Convert to corners
    xt1 = xt - wt / 2
    yt1 = yt - ht / 2
    xt2 = xt + wt / 2
    yt2 = yt + ht / 2
    
    xp1 = xp - wp / 2
    yp1 = yp - hp / 2
    xp2 = xp + wp / 2
    yp2 = yp + hp / 2
    
    # Intersect
    xi1 = K.maximum(xt1, xp1)
    yi1 = K.maximum(yt1, yp1)
    xi2 = K.minimum(xt2, xp2)
    yi2 = K.minimum(yt2, yp2)
    
    inter_area = K.maximum(0.0, xi2 - xi1) * K.maximum(0.0, yi2 - yi1)
    
    b1_area = (xt2 - xt1) * (yt2 - yt1)
    b2_area = (xp2 - xp1) * (yp2 - yp1)
    
    union = b1_area + b2_area - inter_area + K.epsilon()
    
    iou = inter_area / union
    
    return 1.0 - iou

def yolo_loss(y_true, y_pred):
    # y_true/pred: (batch, 10, 10, 8)
    # 0: Conf, 1-4: Box, 5-7: Class

    mask = y_true[..., 0] # (batch, 7, 7) - 1 where obj exists
    noobj_mask = 1 - mask
    
    # Objectness Loss
    conf_loss = K.mean(K.square(y_true[..., 0] - y_pred[..., 0]) * mask) + \
                0.5 * K.mean(K.square(y_true[..., 0] - y_pred[..., 0]) * noobj_mask)

    # Box Loss (IoU Loss)
    # Only calculate where mask=1
    # We flatten to filter by mask
    true_box = tf.boolean_mask(y_true[..., 1:5], mask)
    pred_box = tf.boolean_mask(y_pred[..., 1:5], mask)
    
    box_loss = tf.cond(
        tf.shape(true_box)[0] > 0,
        lambda: K.mean(iou_loss(true_box, pred_box)),
        lambda: 0.0
    )

    # Class Loss (only where mask=1)
    # WEIGHTED LOSS implementation
    # Class 0: with_mask (2759) -> Weight 1.0
    # Class 1: without_mask (523) -> Weight 5.0
    # Class 2: incorrect (94) -> Weight 25.0
    class_weights = tf.constant([1.0, 5.0, 25.0])
    
    # Calculate squared diff per class
    sq_diff = K.square(y_true[..., 5:] - y_pred[..., 5:])
    
    # Apply weights (broadcast)
    weighted_sq_diff = sq_diff * class_weights
    
    # Sum over classes
    class_loss = K.mean(K.sum(weighted_sq_diff, axis=-1) * mask)

    # Total Loss
    return conf_loss + 2.0 * box_loss + 2.0 * class_loss
