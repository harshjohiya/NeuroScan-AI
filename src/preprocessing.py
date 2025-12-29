# data preprocessing and augmentation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class DataPreprocessor:
    
    def __init__(self, config=None):
        self.config = config or {}
        self.datagen = None
    
    def create_augmentation_generator(self, rotation_range=0, zoom_range=0,
                                     width_shift_range=0, height_shift_range=0,
                                     horizontal_flip=True, vertical_flip=False):
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
        if self.datagen is None:
            self.create_augmentation_generator()
        self.datagen.fit(train_data)
    
    def get_generator(self):
        return self.datagen
    
    def normalize_images(self, images):
        return images.astype('float32') / 255.0
    
    def preprocess_single_image(self, img_array, img_size=(150, 150)):
        # convert grayscale to RGB if needed
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        img_array = img_array.reshape(1, img_size[0], img_size[1], 3)
        return img_array
