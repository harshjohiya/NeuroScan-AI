# prediction on new images
import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import BrainTumorCNN
from src.preprocessing import DataPreprocessor


class BrainTumorPredictor:
    
    def __init__(self, model_path, img_size=(150, 150)):
        self.img_size = img_size
        self.class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        self.model = None
        self.preprocessor = DataPreprocessor()
        
        if os.path.exists(model_path):
            cnn = BrainTumorCNN()
            self.model = cnn.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    def predict_image(self, image_path, display=True):
        img = Image.open(image_path)
        img_array = np.array(img.resize(self.img_size))
        
        img_input = self.preprocessor.preprocess_single_image(
            img_array, self.img_size
        )
        
        predictions = self.model.predict_on_batch(img_input)
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[0][predicted_class_idx] * 100
        predicted_class = self.class_names[predicted_class_idx]
        
        if display:
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2f}%',
                     fontsize=14, fontweight='bold')
            plt.show()
        
        print(f"\nPrediction: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"\nAll probabilities:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}: {predictions[0][i]*100:.2f}%")
        
        return {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'all_probabilities': predictions[0]
        }
    
    def predict_batch(self, image_paths):
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
                print(f"Error: {e}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_array(self, img_array, display=True):
        img_input = self.preprocessor.preprocess_single_image(
            img_array, self.img_size
        )
        
        predictions = self.model.predict_on_batch(img_input)
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[0][predicted_class_idx] * 100
        predicted_class = self.class_names[predicted_class_idx]
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain Tumor Prediction')
    parser.add_argument('--model', type=str, default='../models/best_model.h5',
                       help='Path to model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image')
    parser.add_argument('--no-display', action='store_true',
                       help='Skip display')
    
    args = parser.parse_args()
    
    predictor = BrainTumorPredictor(args.model)
    predictor.predict_image(args.image, display=not args.no_display)


if __name__ == '__main__':
    main()
