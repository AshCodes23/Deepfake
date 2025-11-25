import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import albumentations as A

# --- Configuration ---
# Path inside your Linux environment
PROCESSED_DATA_PATH = '/home//processed_data' 
IMG_SIZE = 224
# BATCH_SIZE is set to 8 to prevent GPU Out of Memory errors
BATCH_SIZE = 8 
WARMUP_EPOCHS = 10  # Phase 1: Train only the top layers
FINETUNE_EPOCHS = 5   # Phase 2: Train the whole model
MODEL_SAVE_PATH = 'deepfake_detector_model_final.keras'

def generator_wrapper(generator):
    """A wrapper to make a generator loop indefinitely."""
    while True:
        for batch in generator:
            yield batch

def advanced_augmentation(image):
    """
    Applies advanced augmentations using Albumentations.
    Input: image (numpy array). 
    Note: ImageDataGenerator passes images as float32 if rescale is used, 
    or float32 (0-255) if not rescaled. We handle conversion here.
    """
    # 1. Convert to uint8 (0-255) for Albumentations
    # Check if image is normalized (0-1) or not (0-255)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # 2. Define the Augmentation Pipeline
    transform = A.Compose([
        # Simulate low-quality internet video (JPEG artifacts)
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        
        # Simulate camera noise/grain
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Simulate motion blur or focus issues
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # Randomly block out parts of the face (forces model to look at other features)
        A.CoarseDropout(max_holes=1, max_height=32, max_width=32, 
                        min_holes=1, min_height=16, min_width=16, 
                        fill_value=0, p=0.3),
                        
        # Lighting variations
        A.RandomBrightnessContrast(p=0.5),
    ])

    # 3. Apply Augmentations
    try:
        augmented_image = transform(image=image)['image']
    except Exception as e:
        # Fallback in case of weird errors
        print(f"Augmentation error: {e}")
        augmented_image = image

    # 4. Convert back to float32 (0-1) for the AI model
    return augmented_image.astype(np.float32) / 255.0

def build_model(img_size=(224, 224)):
    """
    Builds the deepfake detection model using Xception as a base.
    """
    # 1. Load the pre-trained Xception model
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3)
    )

    # 2. Freeze the base model layers for the warmup phase
    base_model.trainable = False

    # 3. Add custom classifier head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    # 4. Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model

def main():
    print(f"--- Deepfake Training Pipeline Started ---")
    print(f"Config: Batch Size={BATCH_SIZE}, Warmup={WARMUP_EPOCHS}, Finetune={FINETUNE_EPOCHS}")

    # 1. Setup Data Generators
    
    # TRAIN GENERATOR:
    # We use 'preprocessing_function' for our custom augmentations.
    # IMPORTANT: We DO NOT use 'rescale=1./255' here because 'advanced_augmentation' handles the scaling.
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=advanced_augmentation 
    )

    # VALIDATION GENERATOR:
    # We MUST use rescale here because we are NOT applying the custom function to validation data.
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading Data...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_PATH, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    validation_generator = val_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_PATH, 'val'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"Class indices: {train_generator.class_indices}")

    # 2. Build the model
    model, base_model = build_model(img_size=(IMG_SIZE, IMG_SIZE))

    # 3. Compile for Phase 1
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create the looping generators to prevent "Run out of data" errors
    train_gen_wrapped = generator_wrapper(train_generator)
    val_gen_wrapped = generator_wrapper(validation_generator)

    # --- PHASE 1: WARM-UP (Heads Only) ---
    print(f"\n--- Starting Phase 1: Warm-up Training ({WARMUP_EPOCHS} Epochs) ---")
    history_warmup = model.fit(
        train_gen_wrapped,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_gen_wrapped,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=WARMUP_EPOCHS
    )

    # --- PHASE 2: FINE-TUNING (Full Model) ---
    print(f"\n--- Starting Phase 2: Fine-Tuning ({FINETUNE_EPOCHS} Epochs) ---")
    
    # Un-freeze the base model
    base_model.trainable = True

    # Re-compile with a VERY low learning rate to avoid destroying learned weights
    model.compile(
        optimizer=Adam(learning_rate=1e-5), # 1e-5 = 0.00001
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # model.summary() # Optional: Check trainable params

    history_finetune = model.fit(
        train_gen_wrapped,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_gen_wrapped,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=FINETUNE_EPOCHS
    )

    # 5. Save the final model
    print(f"\n--- Training Complete ---")
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # 6. Plot History
    acc = history_warmup.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history_warmup.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history_warmup.history['loss'] + history_finetune.history['loss']
    val_loss = history_warmup.history['val_loss'] + history_finetune.history['val_loss']

    plt.figure(figsize=(8, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(x=WARMUP_EPOCHS-1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(x=WARMUP_EPOCHS-1, color='gray', linestyle='--', label='Start Fine-Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    
    plt.savefig('training_history_final.png')
    print("Training history plot saved to training_history_final.png")

if __name__ == "__main__":
    main()
