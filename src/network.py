"""
My implementation of Stochastic Gradient Descent (SGD) algorithm. Gradients are calculated using backpropogation.
Referenced sample code provided in /provided_code_old
Updated for further optimization and ensure compatibility with Python 3
"""

# Libraries
import random
import numpy as np


class Network():

    """
    layers_size is a list that encapsulates the number of neurons in each layer.
    [5,4,3,2,1] would mean a five-layer network: an input layer with 5 inputs, 
    an (final) output layer with 1 output, and 3 hidden layers with 4, 3, 2 neurons each
    sandwiched between the former 2.
    The biases and weights are intialised randomly, using default Gaussian distribution
    with mean 0 and SD 1.
    """
    def __init__(self, layers_size):
        self.num_layers = len(layers_size)
        self.sizes = layers_size
        # num here refers to the number of neurons in each layer, and consequently the no. biases 
        self.biases = [np.random.randn(num, 1) for num in layers_size[1:]]
        # Note that each neuron in the hidden layers is connected by num_prev number of weights
        # to num_prev number of neurons in the preceding layer
        # num x num_prev matrix where the values in the jth row represents the weights 
        # connected to the jth neuron in the current output layer
        self.weights = [np.random.randn(num, num_prev) 
                        for num, num_prev in zip(layers_size[1:], layers_size[:len(layers_size)])]
    
    """
    When the inputs are passed (as a (n,1) vector where n is the num of inputs), the output layer is computed, then the function 
    iteratively uses the output layer as input for the computing the next layer until it reaches the final output layer.
    """
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
    test_data: evaluation of network after each epoch to track partial progress
    """
    def SGD(self, training_data, epochs, mini_batch_size, lr, test_data = None):
        training_data = list(training_data) # for shuffling and finding len
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data) # randomly shuffles training data - prevents model from learning the order of training data/unwanted bias
            mini_batches = [training_data[j:j+mini_batch_size] for j in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.update_mini_batch(batch, lr)
            # done with training, evaluate if test_data provided
            if test_data:
                test_data = list(test_data)
                n_test = len(test_data)
                print(f"Epoch {i} completed: {self.evaluate(test_data) / n_test} accuracy")
            else:
                print(f"Epoch {i} completed.")

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
    Backpropagation algorithm. 
    Express change in cost with respect to activation neuron 
    """
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activations = [x] # list representing all activations, layer by layer
        pre_activations = [] # list representing output that has yet to undergo sigmoid, layer by layer

        activation = x
        for b, w in zip(self.biases, self.weights):
            pre = np.dot(w, activation) + b
            pre_activations.append(pre)
            activation = sigmoid(pre)
            activations.append(activation)
            
        # propagate backward
        output_del = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = output_del * sigmoid_prime(pre_activations[-1])
        nabla_w[-1] = np.dot(output_del * sigmoid_prime(pre_activations[-1]), activations[-2].transpose())
       
        for num_layer in range(2, self.num_layers): # input layers have no weights & biases
            pre = pre_activations[-num_layer + 1]
            output_del = np.dot(self.weights[-num_layer+1].transpose(), output_del * sigmoid_prime(pre))
            nabla_b[-num_layer] = output_del * sigmoid_prime(pre_activations[-num_layer])
            nabla_w[-num_layer] = np.dot(output_del * sigmoid_prime(pre_activations[-num_layer]), activations[-num_layer-1].transpose())
        return (nabla_b, nabla_w)
    

    """
    returns the number of test inputs for which the neural network correctly classified
    """
    def evaluate(self, test_data):
        # in the final output layer, the neoron with the highest activation value 
        # is taken to be the classification result. Note that this result is also
        # represented by its index in the output layer
        test_results = [(np.argmax(self.feedForward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    
    def cost_derivative(self, output, actual):
        return 2 * (output-actual)


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

