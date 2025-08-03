import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, 
    UpSampling3D, concatenate, BatchNormalization,
    Activation, Dropout
)

def conv_block(input_tensor, num_filters, dropout_rate=0.2):
    """Create two 3x3x3 conv layers with batch norm and ReLU"""
    x = Conv3D(num_filters, (3, 3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv3D(num_filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x

def simple_unet(input_shape=(128, 128, 128, 4), num_classes=4):
    """3D U-Net model based on Kaggle implementation"""
    
    # Input layer
    inputs = Input(input_shape)
    
    # Encoder path
    c1 = conv_block(inputs, 32)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = conv_block(p1, 64)
    p2 = MaxPooling3D((2, 2, 2))(c2)
    
    c3 = conv_block(p2, 128)
    p3 = MaxPooling3D((2, 2, 2))(c3)
    
    # Bridge
    c4 = conv_block(p3, 256)
    
    # Decoder path
    u5 = UpSampling3D((2, 2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = conv_block(u5, 128)
    
    u6 = UpSampling3D((2, 2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = conv_block(u6, 64)
    
    u7 = UpSampling3D((2, 2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = conv_block(u7, 32)
    
    # Output layer
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c7)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Custom metrics from Kaggle
    dice_coef = tf.keras.metrics.MeanIoU(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', dice_coef]
    )
    
    return model

# For compatibility with your existing code
def build_3d_unet():
    return simple_unet()