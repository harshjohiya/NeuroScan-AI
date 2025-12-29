# CNN model architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
import os


class BrainTumorCNN:
    
    def __init__(self, input_shape=(150, 150, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        model = Sequential()
        
        # conv block 1
        model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same',
                        activation='relu', input_shape=self.input_shape))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # conv block 2
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same',
                        activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        
        # conv block 3
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same',
                        activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.3))
        
        # conv block 4
        model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='Same',
                        activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.3))
        
        # fully connected layers
        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation="softmax"))
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        if self.model is None:
            self.build_model()
        
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
    
    def get_model(self):
        if self.model is None:
            self.build_model()
            self.compile_model()
        return self.model
    
    def summary(self):
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save_model(self, filepath):
        if self.model is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")
    
    def load_model(self, filepath):
        from keras.models import load_model
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
