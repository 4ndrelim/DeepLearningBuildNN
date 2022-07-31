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
        return 0.5 * (a-y)


class CrossEntropyCost(object):
    @staticmethod
    def function(a, y):
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
        self.num_layers = len(layers_size)
        self.sizes = layers_size
        self.weight_initializer()
        self.cost=cost

    def weight_initializer(self):
        self.biases = [np.random.randn(num, 1) for num in self.sizes[1:]]
        # weights are initialised similarly as in network 1.0, albeit now with scaling factor 1/N(in)
        self.weights = [np.random.randn(num, num_prev)/np.sqrt(num_prev)
                        for num, num_prev in zip(self.sizes[1:], self.sizes[:-1])]

    def old_initializer(self):
        self.biases = [np.random.randn(num, 1) for num in self.sizes[1:]]
        self.weights = [np.random.randn(num, num_prev)
                        for num, num_prev in zip(self.sizes[1:], self.sizes[:-1])]

    def feedForward(self, layer):
        for biases, weights in zip(self.biases, self.weights):
            # first calculate the matrix product of weights associated to the current layer applied to the preceding layer
            # add the biases
            # pass the vector into the sigmoid function (the function will be applied on each input)
            layer = sigmoid(np.dot(weights, layer)+biases)
        return layer

    """
    Trains the neural network using mini-batch stochastic gradient descent.
    Monitor the cost and accuracy on either the eval_data or trng_data by setting the correct flags.

    training_data  : list of (x, y) tuples representing inputs and desired outputs
    epochs         : hyperparam that determines the number of complete passes through the training_data set
    mini_batch_size: number of training samples to work through before internal params (weights and biases)
                     are updated 
    lr             : learning_rate
    reg_param      : regularization parameter
    eval_data      : validation data
    
    return         : tuple containing 4 lists, each element of a list represents the per-epoch
                     1. cost on eval_data
                     2. accuracy on eval_data
                     3. cost on trng_data
                     4. accuracy on trng_data
    """
    def SGD(self, training_data, epochs, mini_batch_size, lr, reg_param,
            eval_data=None,
            monitor_eval_cost=False,
            monitor_eval_acc=False,
            monitor_trng_cost=False,
            monitor_trng_acc=False,
            early_stopping=0):
      
        if eval_data:
            eval_data = list(eval_data) # if it is given in zipped form
            n_eval_data = len(eval_data)
            
        # include results data of an untrained network
        eval_cost = []
        eval_acc  = []
        trng_cost = []
        trng_acc  = []

        # implement early stopping
        best_accuracy = 1
        unchangedEpochs = 0
        
        training_data = list(training_data) # if orginally zipped
        n = len(training_data)
        for i in range(epochs+1):
            if i > 0: # run when begins training, epoch 0 considered to be untrained network
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
                trng_acc.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n}")
            if monitor_eval_cost:
                cost = self.total_cost(eval_data, reg_param, convert=True)
                eval_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_eval_acc:
                accuracy = self.accuracy(eval_data)
                eval_acc.append(accuracy)
                print(f"Accuracy on evaluation data: {self.accuracy(eval_data)} / {n_eval_data}")    

            # Implement early stopping
            # note that accuracy here could refer to either
            # a) accuracy on training dataset or
            # b) accuracy on validation dataset
            # latter takes precedence and will be used if both are computed
            
            if early_stopping > 0: # 0 => do not aply early stopping
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    unchangedEpochs = 0
                else:
                    unchangedEpochs += 1
                if unchangedEpochs >= early_stopping:
                    return (eval_cost, eval_acc, trng_cost, trng_acc)
        
       
        return (eval_cost, eval_acc, trng_cost, trng_acc)
            

    """
    Determines the changes to weights and biases using gradient descent then,
    updates weights and biases with back propogation to a single mini batch with L2 regularization:
    a) learning rate, lr
    b) regularization parameter, reg_param
    c) total size of trng_data set, n
    """
    def update_mini_batch(self, mini_batch, lr, reg_param, n):
        total_nabla_biases = [np.zeros(biases.shape) for biases in self.biases]
        total_nabla_weights = [np.zeros(weights.shape) for weights in self.weights]
        for x, y in mini_batch:
            (nabla_biases, nabla_weights) = self.backprop(x, y) # get the change in biases and weights this training input "wishes to see"
            for tnb, nb in zip(total_nabla_biases, nabla_biases):
                tnb += nb
            for tnw, nw in zip(total_nabla_weights, nabla_weights):
                tnw += nw
        size_mini = len(mini_batch)
        
        self.biases  = [b - tnb*(lr/size_mini)
                        for b, tnb in zip(self.biases, total_nabla_biases)]
        self.weights = [(1-lr*(reg_param/n))*w - tnw*(lr/size_mini)
                        for w, tnw in zip(self.weights, total_nabla_weights)]
            
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
        output_del = (self.cost).delta(pre_activations[-1], activations[-1], y)
        nabla_b[-1] = output_del * sigmoid_prime(pre_activations[-1])
        nabla_w[-1] = np.dot(output_del * sigmoid_prime(pre_activations[-1]), activations[-2].transpose())
       
        for num_layer in range(2, self.num_layers): # input layers have no weights & biases
            pre = pre_activations[-num_layer + 1]
            output_del = np.dot(self.weights[-num_layer+1].transpose(), output_del * sigmoid_prime(pre))
            nabla_b[-num_layer] = output_del * sigmoid_prime(pre_activations[-num_layer])
            nabla_w[-num_layer] = np.dot(output_del * sigmoid_prime(pre_activations[-num_layer]), activations[-num_layer-1].transpose())
        return (nabla_b, nabla_w)

    
    def accuracy(self, data, convert=False):
        """
        Return the number of inputs for which the neural
        network outputs the correct result.
        The neural network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.

        <convert> flag:
        a) should be set to False if the data set is
        validation or test data (the usual case),
        b) and to True if the data set is the training data.

        ***From original author***
        The need for this flag arises due to differences in the way the results
        ``y`` are represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.
        
        """
        if convert:
            results = [(np.argmax(self.feedForward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedForward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


    def total_cost(self, data, reg_param, convert=False):
        """
        Return the total cost for the data set.
        The flag <convert> should be set
        a) to False if the data set is the training data (the usual case),
        b) and to True if the data set is the validation or test data.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedForward(x)
            if convert: y = vectorized_result(y)
            cost += (self.cost).function(a, y)/len(data)
        cost += 0.5*(reg_param/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights) 
        return cost

    """
    Saves a trained neural network to the file named <filename>

    Usage:
    In the same program as your model training, simply add the following line:

        model.save(filename)

    where model is an instance of the Network class from network_2.py
    and filename is as suggested
    """
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [weight.tolist() for weight in self.weights],
                "biases": [bias.tolist() for bias in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


"""
Load a trained neural network
Usage:
In the program where you wish to re-create your model, simply add the following lines:

    from network_2.py import load
    network = load(filename)

where filename is the file encapsulating features of the saved model.
Note: make sure filename is in the same directory as the program else import accordingly
"""
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    network = Network(data["sizes"], cost=cost)
    network.weights = [np.array(weight) for weight in data["weights"]]
    network.biases = [np.array(bias) for bias in data["biases"]]
    return network


def vectorized_result(j):
    """
    Return a 10-dimensional unit vector with a 1.0 in the jth position
    and zeroes elsewhere.
    This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

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
    return sigmoid(x) * (1 - sigmoid(x))



