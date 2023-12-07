from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values (0-255) to the 0-1 range
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),          # Flatten the 28x28 images to 1D array of 784 features
    Dense(512, activation='relu'),          # Hidden layer 1 with 512 neurons and ReLU activation
    Dropout(0.2),                           # Dropout layer to reduce overfitting
    Dense(256, activation='relu'),          # Hidden layer 2 with 256 neurons and ReLU activation
    Dropout(0.2),                           # Dropout layer to reduce overfitting
    Dense(10, activation='softmax')         # Output layer with 10 neurons and softmax activation
])

# Compile the model with a reasonable learning rate and loss function
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Function to print training and test accuracy after every epoch
class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs['accuracy']
        test_acc = self.model.evaluate(x_test, y_test, verbose=0)[1]
        print(f'Epoch {epoch+1}: Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

# Create an instance of the accuracy history class
accuracy_history = AccuracyHistory()

# Train the model
model.fit(x_train, y_train,
          epochs=20,                        # Increase epochs for more training
          validation_data=(x_test, y_test),
          callbacks=[accuracy_history])
