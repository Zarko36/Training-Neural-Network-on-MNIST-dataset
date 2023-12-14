# MNIST Dataset Overview
## What is the MNIST Dataset?
The MNIST (Modified National Institute of Standards and Technology) dataset is a large collection of handwritten digits commonly used for training various image processing systems. It's widely used in the field of machine learning and computer vision for benchmarking classification algorithms.

## Dataset Specifications
* Content: The dataset contains 70,000 images of handwritten digits (0 through 9).
* Image Size: Each image is 28x28 pixels, represented in a grayscale format.
* Split: Typically, the dataset is split into 60,000 training images and 10,000 testing images, allowing for robust training and evaluation of models.
## Historical Background
[Here](https://en.wikipedia.org/wiki/MNIST_database) is the wikipage of MNIST dataset please go through it.
We will be using popular deep learning framework tensorflow to complete this activity. First let's load the dataset from tensorflow.keras.datasets and display some instances from the dataset.

![image](https://github.com/Zarko36/Training-Neural-Network-on-MNIST-dataset/assets/74474117/31f00ef3-1da4-44f9-ba95-96d10727222d)
![download](https://github.com/Zarko36/Training-Neural-Network-on-MNIST-dataset/assets/74474117/b2845149-27b6-44d8-96a8-9741458512fb)

# Understanding Neural Networks
Neural networks are a cornerstone of modern artificial intelligence, particularly in the field of deep learning. They are inspired by the structure and function of the human brain, especially in how neurons process and transmit information.

## What is a Neural Network?
A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In essence, it is a system of interconnected units (neurons) that work together to process and analyze data.

## How Do Neural Networks Work?
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
## Learning Resources
To gain a deeper understanding of neural networks, I highly recommend watching this excellent video by [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk) on YouTube. It provides a clear and intuitive explanation of how neural networks function.

[![0](https://github.com/Zarko36/Training-Neural-Network-on-MNIST-dataset/assets/74474117/1e117358-f54a-41d9-ba1a-5596e1946a3b)](https://www.youtube.com/watch?v=aircAruvnKk)

Additionally, for an interactive learning experience, check out the [3Blue1Brown Neural Network Visualization](https://www.3blue1brown.com/lessons/neural-networks). This interactive blog allows you to draw digits and see how a neural network processes your input in real-time, offering a unique perspective on how neural networks make predictions.

# Understanding Neural Network Hyperparameters
In neural networks, hyperparameters are the parameters whose values are set before the learning process begins. These parameters have a significant impact on the training of the network and the final results. Let's discuss some of the essential hyperparameters:

## Number of Layers
* Input Layer: The first layer that receives the input data. Its size is determined by the dimensions of the input data.
* Hidden Layers: Layers between the input and output layers. The number of hidden layers and their size (number of neurons) can greatly affect the network's ability to capture complex patterns.
* Output Layer: The final layer that produces the output. Its size is determined by the number of output classes or values.
## Number of Neurons in a Layer
* The number of neurons in a layer represents the layer's capacity to learn various aspects of the data. More neurons can increase the network's complexity and computational cost.
## Activation Functions
Activation functions introduce non-linear properties to the network, allowing it to learn more complex data patterns.

* ReLU (Rectified Linear Unit): Commonly used in hidden layers, ReLU is defined as  f(x)=max(0,x) . It helps with faster training and mitigating the vanishing gradient problem. Also calculating the gradient of ReLU function is simpler. It is 1 for values of x greater than 0 and 0 otherwise.

![1__vvB81JFM1PGZvYeVI52XQ](https://github.com/Zarko36/Training-Neural-Network-on-MNIST-dataset/assets/74474117/3de8b37f-4d9c-4cb8-8c72-e1be2856a82b)

* Sigmoid: Often used in the output layer for binary classification, it squashes the output between 0 and 1, defined as  f(x)=11+e−x . It's useful for models where we need to predict the probability as an output.

![1_Xu7B5y9gp0iL5ooBj7LtWw](https://github.com/Zarko36/Training-Neural-Network-on-MNIST-dataset/assets/74474117/0660c6c6-35ba-43ce-91a8-ac7dc7eba7c1)

Softmax(xi) = exi/∑jexj

where  xi  is the score (also known as the logit) for class i and the denominator is the sum of exponential scores for all classes. This function ensures that the output probabilities sum up to 1, making it a suitable choice for probabilistic interpretation in classification tasks.

## Learning Rate
* The learning rate defines how quickly or slowly a neural network updates its parameters during training. A too high learning rate can cause the model to converge too quickly to a suboptimal solution, while a too low learning rate can make the training process unnecessarily long.

Understanding and tuning these hyperparameters is crucial for training effective neural networks. Different types of problems may require different configurations for optimal performance.
