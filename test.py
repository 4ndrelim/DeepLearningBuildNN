"""
A program to evaluate the results of constructed neural networks
"""

import mnist_loader
# import libraries
import random
import numpy as np

# import networks
import network
import network_2

# import hyperparameters
from hyperparams import NUM_EPOCHS, LEARNING_RATE, LAYERS, REG_PARAM, MINI_BATCH_SIZE, EARLY_STOPPING


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data, validation_data, test_data = list(training_data), list(validation_data), list(test_data)
# print(list(training_data)[5][0].shape) numpy array! 784 input layer

# Manipulating pseudo-random state
# So you can get the exact same results as I did!
random.seed(123581321)
np.random.seed(123581321)


"""
NOTE:
1. Hyperparameters are specified in hyperparams.py
2. Only the latest version of the network, complemented with additional features is uncommented
3. If you for some reasons you wish to run several networks, re-write the hyperparameters to customise
"""


"""
Neural network 1.0
SGD + backprop using sigmoid function
"""
print("Current: Network 1.0 with basic neural network structure.\n")
net = network.Network(LAYERS)
net.SGD(training_data, NUM_EPOCHS, MINI_BATCH_SIZE, LEARNING_RATE, validation_data = validation_data)
total_test = len(test_data)
results = 100.0 * net.evaluate(test_data)/total_test
print(f"Accuracy on test data: {results}%\n\n")


"""
Neural network 2.0
Similar to 1.0, with the addition of
a) L2 regularization
b) Cross-entropy function
c) Better initialization of weights
d) Early stopping
"""
print(f"Current: Network 2.0 with L2 reg, cross-entropy cost, better initialisation of weights, early stopping of {EARLY_STOPPING} epochs.\n")
net = network_2.Network(LAYERS, cost=network_2.CrossEntropyCost)
net.SGD(training_data, NUM_EPOCHS, MINI_BATCH_SIZE,
        LEARNING_RATE, reg_param = REG_PARAM,
        eval_data = validation_data,
        monitor_eval_cost=False,
        monitor_eval_acc=True,
        monitor_trng_cost=False,
        monitor_trng_acc=False,
        early_stopping=EARLY_STOPPING)
total_test = len(test_data)
results = 100.0 * net.accuracy(test_data, convert=False)/total_test
print(f"Accuracy on test data: {results}%\n\n")




