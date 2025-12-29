"""
Preprocessing module for Brain Tumor Detection and Classification.
Handles data augmentation and preprocessing pipelines.
"""

from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class DataPreprocessor:
    """Handles data preprocessing and augmentation."""
    
    def __init__(self, config=None):
        """
        Initialize the preprocessor.
        
        Args:
            config: Dictionary with preprocessing configuration
        """
        self.config = config or {}
        self.datagen = None
    
    def create_augmentation_generator(self, 
                                     rotation_range=0,
                                     zoom_range=0,
                                     width_shift_range=0,
                                     height_shift_range=0,
                                     horizontal_flip=True,
                                     vertical_flip=False):
        """
        Create an image data generator for augmentation.
        
        Args:
            rotation_range: Rotation range in degrees
            zoom_range: Range for random zoom
            width_shift_range: Width shift range
            height_shift_range: Height shift range
            horizontal_flip: Whether to randomly flip images horizontally
            vertical_flip: Whether to randomly flip images vertically
            
        Returns:
            ImageDataGenerator instance
        """
        self.datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=rotation_range,
            zoom_range=zoom_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip
        )
        return self.datagen
    
    def fit_generator(self, train_data):
        """
        Fit the data generator on training data.
        
        Args:
            train_data: Training data array
        """
        if self.datagen is None:
            self.create_augmentation_generator()
        self.datagen.fit(train_data)
    
    def get_generator(self):
        """Get the configured data generator."""
        return self.datagen
    
    def normalize_images(self, images):
        """
        Normalize image pixel values to [0, 1] range.
        
        Args:
            images: Image array
            
        Returns:
            Normalized image array
        """
        return images.astype('float32') / 255.0
    
    def preprocess_single_image(self, img_array, img_size=(150, 150)):
        """
        Preprocess a single image for prediction.
        
        Args:
            img_array: Image array
            img_size: Target image size
            
        Returns:
            Preprocessed image ready for prediction
        """
        # Ensure correct shape
        if len(img_array.shape) == 2:
            # Convert grayscale to RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Reshape for batch prediction
        img_array = img_array.reshape(1, img_size[0], img_size[1], 3)
        return img_array
