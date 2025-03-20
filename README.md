# mnist_digit_recognition.py
MNIST digit recognition implemented from scratch with NumPy, featuring customizable neuron count and learning rate.
MNIST Handwritten Digit Classification with a Neural Network
Dataset Description:
The dataset used in this project is MNIST (Modified National Institute of Standards and Technology), a well-known dataset consisting of 70,000 grayscale images of handwritten digits (0 to 9). The dataset is divided into:

60,000 training samples
10,000 test samples
Each image is 28x28 pixels, making up a total of 784 features (28*28 pixels) per image.

Approach and Implementation:
In this project, a simple neural network is used to classify the handwritten digits from the MNIST dataset. Here's how the model is structured and how it works:

Data Preprocessing:

The pixel values of the images are normalized to a range between 0 and 1 by dividing by 255 (the maximum pixel value).
The images are then reshaped from a 28x28 2D matrix to a flat 1D vector of size 784, to feed into the fully connected neural network.
The target labels (digit classes) are one-hot encoded into 10 categories (digits 0-9).
Model Architecture:

The neural network consists of two fully connected layers:
The first layer has 128 neurons with a ReLU activation function. This is where the model learns non-linear patterns.
The second layer has 10 neurons corresponding to the 10 possible digits, and a softmax activation function is used to output the probabilities of each class.
The weights and biases are initialized randomly, and the model is trained using gradient descent to minimize the cross-entropy loss between the predicted and actual labels.
Training:

The model is trained for 400 epochs with a learning rate of 0.2. Every 10 epochs, the loss and accuracy are printed out to monitor the model's progress.
During the training process, backpropagation is used to adjust the weights based on the gradient of the loss function.
Observations and Challenges:
Overly Large Number of Neurons:
Increasing the number of neurons in the network (e.g., 64 in the first layer) can cause issues such as overfitting and slower convergence during training. Too many neurons may lead to an excessively complex model that is harder to train, and in many cases, doesn't result in significant improvements in accuracy.
Learning Rate:
While increasing the learning rate can sometimes speed up training, it has diminishing returns. Beyond a certain threshold, a high learning rate can lead to instability, making the model fail to converge. In this case, a learning rate of 0.2 is used, but increasing it further would not significantly improve accuracy.

import tensorflow as tf  # Import TensorFlow library
from tensorflow.keras.datasets import mnist  # Import MNIST dataset from Keras
from tensorflow.keras.utils import to_categorical  # Import function for one-hot encoding of labels

# Load MNIST dataset and split into training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train, x_test = x_train / 255, x_test / 255

# Reshape the images to a flat vector of size 28*28 (784) for each image
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Convert the labels into one-hot encoded vectors (10 categories)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

import numpy as np  # Import NumPy library for numerical operations

# Initialize weights for the neural network (random values for the first layer)
w1 = np.random.randn(28 * 28, 128)  # Weights for the first layer (784 input nodes, 128 hidden nodes)
b1 = np.zeros((1, 128))  # Bias for the first layer (128 hidden nodes)

# Initialize weights for the second layer (128 hidden nodes, 10 output nodes)
w2 = np.random.randn(128, 10)  # Weights for the second layer (128 hidden nodes, 10 output nodes)
b2 = np.zeros((1, 10))  # Bias for the second layer (10 output nodes)

epochs = 400  # Set the number of training epochs (iterations)
for epoch in range(epochs):  # Loop through each epoch
    # Forward pass for the first layer (linear transformation + activation function)
    z1 = x_train @ w1 + b1  # Linear combination of input and weights, plus bias
    z1 = np.maximum(0, z1)  # ReLU activation function (set all negative values to 0)

    # Forward pass for the second layer (linear transformation)
    z2 = z1 @ w2 + b2  # Linear combination of hidden layer output and second layer weights, plus bias

    # Softmax function to calculate probabilities for each class
    y_pred = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)  # Softmax activation

    # Compute the loss (cross-entropy loss)
    loss = -np.sum(y_train * np.log(y_pred)) / y_train.shape[0]  # Compute the average cross-entropy loss

    # Backpropagation to compute gradients (error propagation from output to input)
    dz2 = y_pred - y_train  # Gradient of loss with respect to z2
    dw2 = np.dot(z1.T, dz2) / y_train.shape[0]  # Gradient of loss with respect to w2
    db2 = np.sum(dz2, axis=0, keepdims=True) / y_train.shape[0]  # Gradient of loss with respect to b2

    # Backpropagation to the first layer
    dz1 = np.dot(dz2, w2.T) * (z1 > 0)  # Gradient of loss with respect to z1 (ReLU derivative)
    dw1 = np.dot(x_train.T, dz1) / x_train.shape[0]  # Gradient of loss with respect to w1
    db1 = np.sum(dz1, axis=0, keepdims=True) / x_train.shape[0]  # Gradient of loss with respect to b1

    learning_rate = 0.2  # Set the learning rate (step size for gradient descent)

    # Update the weights and biases using gradient descent
    w1 -= learning_rate * dw1  # Update weights for the first layer
    b1 -= learning_rate * db1  # Update biases for the first layer
    w2 -= learning_rate * dw2  # Update weights for the second layer
    b2 -= learning_rate * db2  # Update biases for the second layer

    # Print the loss and accuracy every 10 epochs
    if epoch % 10 == 0:
        y_pred_labels = np.argmax(y_pred, axis=1)  # Get predicted labels by choosing the max probability
        y_true_labels = np.argmax(y_train, axis=1)  # Get true labels (original labels)
        accuracy = np.mean(y_pred_labels == y_true_labels)  # Calculate accuracy
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")  # Print epoch info




#Python 
#NeuralNetwork 
#DeepLearning 
#MachineLearning 
#MNIST 
#NumPy 
#AI 
#DataScience 
#HandwrittenDigitRecognition 
#FeedforwardNetwork
