import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

class DiseaseDataPreprocessor:
    def __init__(self, image_size=(224, 224), color_mode="rgb"):
        """
        image_size: tuple of (height, width)
        color_mode: "rgb" for 3 channels, "grayscale" for 1 channel
        """
        self.image_size = image_size
        self.color_mode = color_mode
        self.class_indices = None

    def setup_data_generators(self, data_path, batch_size=32, validation_split=0.2):
        """Setup training and validation generators with augmentation"""
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )

        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            data_path,
            target_size=self.image_size,
            color_mode=self.color_mode,  # <-- Explicit color mode
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        val_generator = val_datagen.flow_from_directory(
            data_path,
            target_size=self.image_size,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        # Save class indices
        self.class_indices = train_generator.class_indices

        print(f"✅ Loaded {train_generator.samples} training samples")
        print(f"✅ Loaded {val_generator.samples} validation samples")
        print(f"✅ Number of classes: {len(self.class_indices)}")

        return train_generator, val_generator

    def preprocess_single_image(self, image_path):
        """Preprocess a single image for prediction"""
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=self.image_size, color_mode=self.color_mode
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Batch dimension
        img_array /= 255.0  # Normalize

        return img_array

    def save_class_indices(self, path):
        """Save class indices to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.class_indices, f, indent=4)

    def load_class_indices(self, path):
        """Load class indices from JSON file"""
        with open(path, 'r') as f:
            self.class_indices = json.load(f)
        return self.class_indices


if __name__ == "__main__":
    preprocessor = DiseaseDataPreprocessor(color_mode="rgb")
    print("✅ Disease data preprocessor ready!")
