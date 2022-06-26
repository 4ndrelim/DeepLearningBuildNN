"""
Improved version of network.py,
improving the SGD-based (Stochastic Gradient Descent) with
a) cross-entropy cost function,
b) regularisation,
c) better initialisation of weights
"""

# import libraries
import json
import random
import sys
import numpy as np


# Cost functions
class QuadraticCost(object):
    @staticmethod
    def function(a, y):
        return (a-y)**2
    
    @staticmethod
    def delta(z, a, y):
        return 0.5 * (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        # np.nan_to_num is used to ensure numerical
        # stability. The np.nan_to_num ensures NaN 
        # values are converted to (0.0).
        
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)


#Network 2.0
class Network(object):

    def __init__(self, layers_size, cost=CrossEntropyCost):
        """
        Similar to previous, layers_size is a list that encapsulates the number of neurons in each layer.
        The biases and weights for the network are initialized using default_weight_initializer.

        """
        self.num_layers = len(sizes)
        self.sizes = layers_size
        self.default_weight_initializer()
        self.cost=cost

    def weight_initializer(self):
        self.biases = [np.random.randn(num, 1) for num in self.sizes[1:]]
        # weights are initialised similarly as in network 1.0, albeit now with scaling factor 1/N(in)
        self.weights = [np.random.randn(num, num_prev)/np.sqrt(num_prev)
                        for num, num_prev in zip(self.sizes[1:], self.sizes[:len(self.sizes)-1])]

    def old_initializer(self):
        self.biases = [np.random.randn(num, 1) for num in self.sizes[1:]]
        self.weights = [np.random.randn(num, num_prev)
                        for num, num_prev in zip(self.sizes[1:], self.sizes[:len(self.sizes)-1])]

    def feedForward(self, layer):
        for biases, weights in zip(self.biases, self.weights):
            # first calculate the matrix product of weights associated to the current layer applied to the preceding layer
            # add the biases
            # pass the vector into the sigmoid function (the function will be applied on each input)
            layer = sigmoid(np.dot(weights, layer) + biases)
        return layer

    """
    Trains the neural network using mini-batch stochastic gradient descent.
    training_data: list of (x, y) tuples representing inputs and desired outputs
    epochs: hyperparam that determines the number of complete passes through the training_data set
    mini_batch_size: number of training samples to work through before internal params (weights and biases) are updated 
    lr: learning_rate
    reg_param: regularization parameter
    eval_data: validation data
    
    """
    def SGD(self, training_data, epochs, mini_batch_size, lr, reg_param,
            eval_data=None,
            monitor_eval_cost=False,
            monitor_eval_acc=False,
            minitor_trng_cost=False,
            monitor_trng_acc=False):
        
        if eval_data:
            n_eval_data = len(eval_data)
        eval_cost, eval_acc = [], []
        trng_cost, trng_acc = [], []
        
        training_data = list(training_data) # for shuffling and finding len
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data) # randomly shuffles training data - prevents model from learning the order of training data/unwanted bias
            mini_batches = [training_data[j:j+mini_batch_size] for j in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.update_mini_batch(batch, lr, reg_param, len(training_data))
            print(f"Epoch {i} completed.")
            
            if monitor_trng_cost:
                cost = self.total_cost(training_data, reg_param)
                trng_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_trng_acc:
                accuracy = self.accuracy(training_data, convert=True)
                trng_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n}")
            if monitor_eval_cost:
                cost = self.total_cost(eval_data, reg_param, convert=True)
                eval_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_eval_acc:
                accuracy = self.accuracy(eval_data)
                eval_acc.append(accuracy)
                print(f"Accuracy on evaluation data: {self.accuracy(eval_data)} / {n_data}")    
        return eval_cost, eval_acc, \
            trng_cost, trng_acc
            

    """
    Determines the changes to weights and biases using gradient descent then,
    updates weights and biases with back propogation to a single mini batch with learning rate lr.
    """
    def update_mini_batch(self, mini_batch, lr):
        total_nabla_biases = [np.zeros(biases.shape) for biases in self.biases]
        total_nabla_weights = [np.zeros(weights.shape) for weights in self.weights]
        for (x, y) in mini_batch:
            (nabla_biases, nabla_weights) = self.backprop(x, y) # get the change in biases and weights this training input "wishes to see"
            for tnb, nb in zip(total_nabla_biases, nabla_biases):
                tnb += nb
            for tnw, nw in zip(total_nabla_weights, nabla_weights):
                tnw += nw
        size_mini = len(mini_batch)
        for b, tnb in zip(self.biases, total_nabla_biases):
            b -= lr * tnb / size_mini
        for w, tnw in zip(self.weights, total_nabla_weights):
            w -= lr * tnw / size_mini

    
        

"""
Function that squishes inputs to a value between 0 and 1
Very negative inputs tend to 0
Very positive inputs tend to 1
"""
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

"""
derivative function of sigmoid
"""
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))



