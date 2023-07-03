import os
import numpy as np
import tensorflow as tf

# Defining a class called NeuralNetwork
class NeuralNetwork:
    
    # Initializing an instance of the class
    def __init__(self, df, y):
        
        # Setting the random seed for numpy
        np.random.seed(42)
        
        # Setting the random seed for TensorFlow
        tf.random.set_seed(42)
        
        # Defining a sequential model using the Keras Sequential API
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(300, activation='relu', input_shape=(df.shape[1],)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Compiling the model with the Adam optimizer and mean squared error loss
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Training the model
        self.train(df, y)

    # Method to train the neural network model
    def train(self, df, y):
        
        # Fitting the model to the training data with a specified number of epochs, batch size, and verbosity level
        self.model.fit(df, y, epochs=50, batch_size=8, verbose=0)
    
    # Method to save the trained model
    def save_model(self):
        
        # Defining the directory path for the model
        model_dir = os.path.join("..", "model")
        
        # Defining the file path for the model
        file_path = os.path.join(model_dir, "model.h5")
        
        # Creating the directory for the model if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Saving the model to the specified file path using the Keras save_model function
        tf.keras.models.save_model(self.model, file_path)
        
        # Printing a success message
        print("Model saved successfully.")

