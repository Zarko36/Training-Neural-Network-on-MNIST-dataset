# What is the MNIST Dataset?
The MNIST (Modified National Institute of Standards and Technology) dataset is a large collection of handwritten digits commonly used for training various image processing systems. It's widely used in the field of machine learning and computer vision for benchmarking classification algorithms.

# Dataset Specifications
* Content: The dataset contains 70,000 images of handwritten digits (0 through 9).
* Image Size: Each image is 28x28 pixels, represented in a grayscale format.
* Split: Typically, the dataset is split into 60,000 training images and 10,000 testing images, allowing for robust training and evaluation of models.
# Historical Background
[Here](https://en.wikipedia.org/wiki/MNIST_database) is the wikipage of MNIST dataset please go through it.
We will be using popular deep learning framework tensorflow to complete this activity. First let's load the dataset from tensorflow.keras.datasets and display some instances from the dataset.

![image](https://github.com/Zarko36/Training-Neural-Network-on-MNIST-dataset/assets/74474117/31f00ef3-1da4-44f9-ba95-96d10727222d)
![download](https://github.com/Zarko36/Training-Neural-Network-on-MNIST-dataset/assets/74474117/b2845149-27b6-44d8-96a8-9741458512fb)

# Understanding Neural Networks
Neural networks are a cornerstone of modern artificial intelligence, particularly in the field of deep learning. They are inspired by the structure and function of the human brain, especially in how neurons process and transmit information.

# What is a Neural Network?
A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In essence, it is a system of interconnected units (neurons) that work together to process and analyze data.

# How Do Neural Networks Work?
Neural networks operate on layers of neurons. Each neuron in a layer is connected to neurons in the previous and next layers. These connections have associated weights and biases, which are adjusted during the training process.

![0__SH7tsNDTkGXWtZb](https://github.com/Zarko36/Training-Neural-Network-on-MNIST-dataset/assets/74474117/a1e69d6d-ab6d-41f7-9881-fc4dffc0e34c)

* Input Layer: This is where the network receives its input data.

* Hidden Layers: These layers perform computations using activated weights and are the main computational engine of the neural network.

* Output Layer: The final layer that outputs the prediction or classification.

The process of a neural network can be summarized in three key steps:

* Feedforward: Input data is passed through the layers of the network. Each neuron applies a weighted sum on the inputs, adds a bias, and then passes it through an activation function.

* Backpropagation: The network compares the output it produced with the actual desired output and calculates the error.

* Weight Adjustment: Using algorithms like Gradient Descent, the network adjusts its weights and biases to minimize the error.

Mathematically, the operation in each neuron can be represented as:

y=f(∑i(wi⋅xi)+b) 

Where:
* y  is the output.
* f  is the activation function.
* wi  are the weights.
* xi  are the inputs.
* b  is the bias.
# Learning Resources
To gain a deeper understanding of neural networks, I highly recommend watching this excellent video by [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) on YouTube. It provides a clear and intuitive explanation of how neural networks function.
