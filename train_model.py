import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# Load and preprocess the MNIST dataset
def load_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
    
    return (train_images, train_labels), (test_images, test_labels)

# Build a simple CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Train the model and save it
def train_and_save_model():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    model = build_model()
    model.fit(train_images, train_labels, epochs=5)
    
    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save the trained model
    model.save('mnist_cnn_model.h5')
    print("Model saved as mnist_cnn_model.h5")

if __name__ == "__main__":
    train_and_save_model()
