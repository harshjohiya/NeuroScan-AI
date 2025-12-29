"""
Evaluation module for Brain Tumor Detection and Classification.
Handles model evaluation and performance metrics.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import BrainTumorDataLoader
from src.model import BrainTumorCNN


class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, model_path=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        cnn = BrainTumorCNN()
        self.model = cnn.load_model(model_path)
    
    def evaluate(self, test_data, test_labels):
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test images
            test_labels: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Load a model first.")
        
        # Get predictions
        predictions = self.model.predict(test_data)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        
        # Evaluate with Keras
        loss, keras_accuracy = self.model.evaluate(test_data, test_labels, verbose=0)
        
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes
        }
    
    def plot_confusion_matrix(self, cm, classes, normalize=False, 
                            title='Confusion Matrix', cmap=plt.cm.Blues,
                            save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            classes: List of class names
            normalize: Whether to normalize the matrix
            title: Plot title
            cmap: Color map
            save_path: Path to save the plot
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, test_data, test_labels, save_dir='../Results'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            test_data: Test images
            test_labels: Test labels
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Evaluate
        results = self.evaluate(test_data, test_labels)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            results['true_classes'],
            results['predicted_classes'],
            target_names=self.class_names
        ))
        
        # Save classification report
        report = classification_report(
            results['true_classes'],
            results['predicted_classes'],
            target_names=self.class_names
        )
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(results['true_classes'], results['predicted_classes'])
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            cm, self.class_names,
            title='Confusion Matrix',
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # Plot normalized confusion matrix
        self.plot_confusion_matrix(
            cm, self.class_names,
            normalize=True,
            title='Normalized Confusion Matrix',
            save_path=os.path.join(save_dir, 'confusion_matrix_normalized.png')
        )
        
        print(f"\nEvaluation report saved to {save_dir}")
        return results


def main():
    """Main evaluation script."""
    # Load test data
    print("Loading test data...")
    data_loader = BrainTumorDataLoader(img_size=(150, 150))
    test_data, test_labels = data_loader.load_test_data('../data/raw/Testing')
    
    # Initialize evaluator
    evaluator = ModelEvaluator('../models/best_model.h5')
    
    # Generate report
    evaluator.generate_report(test_data, test_labels)


if __name__ == '__main__':
    main()
