"""
Model architecture module for Brain Tumor Detection and Classification.
Defines the CNN model architecture.
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
import os


class BrainTumorCNN:
    """CNN model for brain tumor classification."""
    
    def __init__(self, input_shape=(150, 150, 3), num_classes=4):
        """
        Initialize the model.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classification classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """
        Build the CNN architecture.
        
        Returns:
            Keras Sequential model
        """
        model = Sequential()
        
        # First convolutional block
        model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same',
                        activation='relu', input_shape=self.input_shape))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second convolutional block
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same',
                        activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        
        # Third convolutional block
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same',
                        activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.3))
        
        # Fourth convolutional block
        model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='Same',
                        activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.3))
        
        # Dense layers
        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation="softmax"))
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        if self.model is None:
            self.build_model()
        
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
    
    def get_model(self):
        """Get the compiled model."""
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        """
        Save the model to a file.
        
        Args:
            filepath: Path where to save the model
        """
        if self.model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Build the model first.")
    
    def load_model(self, filepath):
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        from keras.models import load_model
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
