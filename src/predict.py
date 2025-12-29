"""
Prediction module for Brain Tumor Detection and Classification.
Handles inference on new images.
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import BrainTumorCNN
from src.preprocessing import DataPreprocessor


class BrainTumorPredictor:
    """Handles predictions on new images."""
    
    def __init__(self, model_path, img_size=(150, 150)):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            img_size: Target image size for prediction
        """
        self.img_size = img_size
        self.class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        self.model = None
        self.preprocessor = DataPreprocessor()
        
        # Load model
        if os.path.exists(model_path):
            cnn = BrainTumorCNN()
            self.model = cnn.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def predict_image(self, image_path, display=True):
        """
        Predict the class of a single image.
        
        Args:
            image_path: Path to the image file
            display: Whether to display the image with prediction
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        img = Image.open(image_path)
        img_array = np.array(img.resize(self.img_size))
        
        # Preprocess for prediction
        img_input = self.preprocessor.preprocess_single_image(
            img_array, self.img_size
        )
        
        # Make prediction
        predictions = self.model.predict_on_batch(img_input)
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[0][predicted_class_idx] * 100
        predicted_class = self.class_names[predicted_class_idx]
        
        # Display results
        if display:
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2f}%',
                     fontsize=14, fontweight='bold')
            plt.show()
        
        # Print results
        print(f"\nPrediction Results:")
        print(f"Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"\nAll class probabilities:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: {predictions[0][i]*100:.2f}%")
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'all_probabilities': predictions[0]
        }
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for img_path in image_paths:
            print(f"\nProcessing: {img_path}")
            try:
                result = self.predict_image(img_path, display=False)
                results.append({
                    'image_path': img_path,
                    **result
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_array(self, img_array, display=True):
        """
        Predict from a numpy array.
        
        Args:
            img_array: Image as numpy array
            display: Whether to display the image with prediction
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        img_input = self.preprocessor.preprocess_single_image(
            img_array, self.img_size
        )
        
        # Make prediction
        predictions = self.model.predict_on_batch(img_input)
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[0][predicted_class_idx] * 100
        predicted_class = self.class_names[predicted_class_idx]
        
        # Display results
        if display:
            plt.figure(figsize=(8, 6))
            plt.imshow(img_array)
            plt.axis('off')
            plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2f}%',
                     fontsize=14, fontweight='bold')
            plt.show()
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'all_probabilities': predictions[0]
        }


def main():
    """Main prediction script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain Tumor Classification Prediction')
    parser.add_argument('--model', type=str, default='../models/best_model.h5',
                       help='Path to the trained model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to the image to predict')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display the image')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = BrainTumorPredictor(args.model)
    
    # Make prediction
    predictor.predict_image(args.image, display=not args.no_display)


if __name__ == '__main__':
    main()
