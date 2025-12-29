"""
Data loading module for Brain Tumor Detection and Classification.
Handles loading and organizing MRI image data from dataset directories.
"""

import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder


class BrainTumorDataLoader:
    """Loads and preprocesses brain tumor MRI images."""
    
    def __init__(self, img_size=(150, 150)):
        """
        Initialize the data loader.
        
        Args:
            img_size: Tuple of (width, height) for resizing images
        """
        self.img_size = img_size
        self.classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        self.encoder = OneHotEncoder()
        self.encoder.fit([[0], [1], [2], [3]])
    
    def get_class_name(self, number):
        """
        Convert class index to class name.
        
        Args:
            number: Class index (0-3)
            
        Returns:
            Class name string
        """
        if 0 <= number < len(self.classes):
            return self.classes[number]
        return None
    
    def load_data(self, data_path, verbose=True):
        """
        Load images from a directory structure.
        
        Args:
            data_path: Path to directory containing class subdirectories
            verbose: Whether to print loading progress
            
        Returns:
            Tuple of (data_array, label_array)
        """
        data = []
        labels = []
        
        for index, class_dir in enumerate(os.listdir(data_path)):
            subdir = os.path.join(data_path, class_dir)
            if not os.path.isdir(subdir):
                continue
                
            for file in os.listdir(subdir):
                img_path = os.path.join(subdir, file)
                try:
                    img = Image.open(img_path)
                    img_resized = img.resize(self.img_size)
                    img_array = np.array(img_resized)
                    
                    data.append(img_array)
                    labels.append(self.encoder.transform([[index]]).toarray())
                except Exception as e:
                    if verbose:
                        print(f"Error loading {img_path}: {e}")
            
            if verbose:
                print(f"Loaded class: {self.get_class_name(index)} ({class_dir})")
        
        data_array = np.array(data)
        labels_array = np.array(labels).reshape(len(labels), 4)
        
        if verbose:
            print(f"Data shape: {data_array.shape}")
            print(f"Labels shape: {labels_array.shape}")
        
        return data_array, labels_array
    
    def load_train_data(self, train_path):
        """Load training data."""
        return self.load_data(train_path, verbose=True)
    
    def load_test_data(self, test_path):
        """Load testing data."""
        return self.load_data(test_path, verbose=True)
