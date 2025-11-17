import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- Configuration ---
PROCESSED_DATA_PATH = './processed_data'
IMG_SIZE = 224
BATCH_SIZE = 32 # You may need to lower this if you run out of GPU memory
EPOCHS = 10
MODEL_SAVE_PATH = 'deepfake_detector_model.h5'

def build_model(img_size=(224, 224)):
    """
    Builds the deepfake detection model using Xception as a base.
    """
    # 1. Load the pre-trained Xception model
    base_model = Xception(
        weights='imagenet',
        include_top=False,  # Do not include the final ImageNet classifier layer
        input_shape=(img_size[0], img_size[1], 3)
    )

    # 2. Freeze the base model layers
    base_model.trainable = False

    # 3. Add our custom classifier head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Final output layer: 1 neuron, sigmoid activation (for binary classification)
    predictions = Dense(1, activation='sigmoid')(x)

    # 4. Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def main():
    # 1. Setup Data Generators
    # Rescale pixel values from [0, 255] to [0, 1] as Xception expects
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_PATH, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'  # 'fake' and 'real' will be 0 and 1
    )

    validation_generator = datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_PATH, 'val'),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    print(f"Class indices: {train_generator.class_indices}") # Should be {'fake': 0, 'real': 1} or vice-versa

    # 2. Build the model
    model = build_model(img_size=(IMG_SIZE, IMG_SIZE))

    # 3. Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # 4. Train the model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # 5. Save the final model
    print(f"\n--- Training Complete ---")
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Optional: Plot training history
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_accuracy.png')
    
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')

if __name__ == "__main__":
    # Import os for the generators
    import os
    main()