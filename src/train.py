"""
Training module for Brain Tumor Detection and Classification.
Handles model training with data augmentation and callbacks.
"""

import sys
import os
import yaml
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import BrainTumorDataLoader
from src.preprocessing import DataPreprocessor
from src.model import BrainTumorCNN


class ModelTrainer:
    """Handles model training process."""
    
    def __init__(self, config_path=None):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.model = None
        self.history = None
    
    def load_config(self, config_path):
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'model': {
                'input_shape': [150, 150, 3],
                'num_classes': 4,
                'learning_rate': 0.001
            },
            'training': {
                'epochs': 40,
                'batch_size': 40
            },
            'data': {
                'train_path': '../Dataset/Training',
                'test_path': '../Dataset/Testing',
                'img_size': [150, 150]
            }
        }
    
    def train(self, train_data, train_labels, val_data, val_labels, 
              epochs=None, batch_size=None):
        """
        Train the model.
        
        Args:
            train_data: Training images
            train_labels: Training labels
            val_data: Validation images
            val_labels: Validation labels
            epochs: Number of epochs (overrides config)
            batch_size: Batch size (overrides config)
            
        Returns:
            Training history
        """
        # Get parameters
        epochs = epochs or self.config['training']['epochs']
        batch_size = batch_size or self.config['training']['batch_size']
        
        # Initialize model
        model_config = self.config['model']
        cnn = BrainTumorCNN(
            input_shape=tuple(model_config['input_shape']),
            num_classes=model_config['num_classes']
        )
        cnn.compile_model(learning_rate=model_config['learning_rate'])
        self.model = cnn.get_model()
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.create_augmentation_generator(
            rotation_range=0,
            zoom_range=0,
            width_shift_range=0,
            height_shift_range=0,
            horizontal_flip=True,
            vertical_flip=False
        )
        preprocessor.fit_generator(train_data)
        datagen = preprocessor.get_generator()
        
        # Train model
        print(f"Starting training for {epochs} epochs...")
        self.history = self.model.fit_generator(
            datagen.flow(train_data, train_labels, batch_size=batch_size),
            epochs=epochs,
            validation_data=(val_data, val_labels),
            steps_per_epoch=train_data.shape[0] // batch_size
        )
        
        print("Training completed!")
        return self.history
    
    def save_model(self, filepath='../models/brain_tumor_model.h5'):
        """
        Save the trained model.
        
        Args:
            filepath: Path where to save the model
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Train the model first.")
    
    def plot_training_history(self, save_path=None):
        """
        Plot training and validation loss/accuracy.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            print("No training history available.")
            return
        
        # Plot loss
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def main():
    """Main training script."""
    # Initialize components
    config_path = '../configs/config.yaml'
    trainer = ModelTrainer(config_path)
    
    # Load data
    print("Loading data...")
    data_loader = BrainTumorDataLoader(
        img_size=tuple(trainer.config['data']['img_size'])
    )
    
    train_data, train_labels = data_loader.load_train_data(
        trainer.config['data']['train_path']
    )
    test_data, test_labels = data_loader.load_test_data(
        trainer.config['data']['test_path']
    )
    
    # Train model
    trainer.train(train_data, train_labels, test_data, test_labels)
    
    # Save model
    trainer.save_model('../models/best_model.h5')
    
    # Plot results
    trainer.plot_training_history('../Results/training_history.png')


if __name__ == '__main__':
    main()
